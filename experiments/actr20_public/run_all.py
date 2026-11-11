"""One-command, process-isolated ACTR public experiment.

The benchmark and every model audit run in a fresh Python interpreter.  This is
not an implementation detail: repeated histogram-boosting fits can retain
process-local native thread-pool state on some BLAS/OpenMP combinations.  The
orchestrator exchanges only files, writes one log per stage, and commits an
atomic run-status artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from audit_plan import AUDIT_NAMES


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = "1"
    environment["PYTHONHASHSEED"] = "0"
    return environment


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:  # pragma: no cover - exercised on Windows CI/users
        process.terminate()


def _run_stage(
    *,
    name: str,
    command: Sequence[str],
    log_path: Path,
    environment: dict[str, str],
    deadline: float,
    heartbeat_seconds: float,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[TimeSafe] starting {name}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            list(command),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=(os.name != "nt"),
        )
        try:
            next_heartbeat = time.monotonic() + heartbeat_seconds
            while process.poll() is None:
                now = time.monotonic()
                if now >= deadline:
                    _terminate(process)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                    raise TimeoutError(
                        f"Stage {name!r} exceeded the experiment timeout.\n{tail}"
                    )
                if now >= next_heartbeat:
                    print(f"[TimeSafe] {name} still running", flush=True)
                    next_heartbeat = now + heartbeat_seconds
                time.sleep(min(0.5, heartbeat_seconds))
        except BaseException:
            _terminate(process)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            raise

    if process.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
        raise RuntimeError(
            f"Stage {name!r} failed with exit code {process.returncode}.\n{tail}"
        )
    print(f"[TimeSafe] completed {name}", flush=True)


def _prepare_output(output: Path, *, overwrite: bool) -> None:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output}. "
                "Use --overwrite or choose a fresh directory."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir()
    (output / "audit_workers").mkdir()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actr-root", required=True)
    parser.add_argument("--output-dir", default="results/generated")
    parser.add_argument(
        "--total-timeout-seconds",
        "--timeout-seconds",
        dest="total_timeout_seconds",
        type=int,
        default=21600,
        help="Maximum wall-clock time for the complete experiment (default: 6 hours).",
    )
    parser.add_argument(
        "--stage-timeout-seconds",
        type=int,
        default=3600,
        help="Maximum wall-clock time for each isolated stage (default: 1 hour).",
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (
        args.total_timeout_seconds <= 0
        or args.stage_timeout_seconds <= 0
        or args.heartbeat_seconds <= 0
    ):
        parser.error("total timeout, stage timeout, and heartbeat values must be positive")

    here = Path(__file__).resolve().parent
    actr_root = Path(args.actr_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    _prepare_output(output, overwrite=args.overwrite)

    status_file = output / "run_status.json"
    started_wall = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started = time.monotonic()
    total_deadline = started + args.total_timeout_seconds

    def stage_deadline() -> float:
        return min(
            total_deadline,
            time.monotonic() + args.stage_timeout_seconds,
        )
    environment = _worker_environment()
    payload = {
        "status": "running",
        "exit_code": None,
        "started_at": started_wall,
        "finished_at": None,
        "python": sys.executable,
        "total_timeout_seconds": args.total_timeout_seconds,
        "stage_timeout_seconds": args.stage_timeout_seconds,
    }
    _atomic_json(status_file, payload)

    try:
        _run_stage(
            name="benchmark reproduction",
            command=[
                sys.executable,
                "-u",
                str(here / "run_benchmark.py"),
                "--actr-root",
                str(actr_root),
                "--output-dir",
                str(output),
            ],
            log_path=output / "logs" / "benchmark.log",
            environment=environment,
            deadline=stage_deadline(),
            heartbeat_seconds=args.heartbeat_seconds,
        )
        benchmark_config = json.loads(
            (output / "benchmark_config.json").read_text(encoding="utf-8")
        )
        run_id = benchmark_config.get("experiment_run_id")
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError("Benchmark stage did not create an experiment_run_id.")
        payload.update(
            {
                "experiment_run_id": run_id,
                "implementation_sha256": benchmark_config.get(
                    "implementation_sha256"
                ),
                "benchmark_sha256": benchmark_config.get("data_sha256"),
            }
        )
        _atomic_json(status_file, payload)

        for position, audit in enumerate(AUDIT_NAMES, start=1):
            _run_stage(
                name=f"audit {position}/{len(AUDIT_NAMES)}: {audit}",
                command=[
                    sys.executable,
                    "-u",
                    str(here / "audit_worker.py"),
                    "--audit",
                    audit,
                    "--actr-root",
                    str(actr_root),
                    "--output",
                    str(output / "audit_workers" / f"{audit}.json"),
                    "--run-id",
                    run_id,
                ],
                log_path=output / "logs" / f"audit_{audit}.log",
                environment=environment,
                deadline=stage_deadline(),
                heartbeat_seconds=args.heartbeat_seconds,
            )

        _run_stage(
            name="audit aggregation",
            command=[
                sys.executable,
                "-u",
                str(here / "run_audits.py"),
                "--combine-only",
                "--output-dir",
                str(output),
            ],
            log_path=output / "logs" / "combine_audits.log",
            environment=environment,
            deadline=stage_deadline(),
            heartbeat_seconds=args.heartbeat_seconds,
        )
        report = output / "ACTR20_TIMESAFE_EXPERIMENT.md"
        _run_stage(
            name="evidence report",
            command=[
                sys.executable,
                "-u",
                str(here / "assemble_report.py"),
                "--results-dir",
                str(output),
                "--output",
                str(report),
            ],
            log_path=output / "logs" / "assemble_report.log",
            environment=environment,
            deadline=stage_deadline(),
            heartbeat_seconds=args.heartbeat_seconds,
        )
        if not report.is_file():
            raise RuntimeError("Evidence report is missing after successful stages.")
        payload.update(
            {
                "status": "success",
                "exit_code": 0,
                "finished_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "report": str(report),
            }
        )
        _atomic_json(status_file, payload)
        print(report)
    except BaseException as exc:
        payload.update(
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "exit_code": 1,
                "finished_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _atomic_json(status_file, payload)
        raise


if __name__ == "__main__":
    main()
