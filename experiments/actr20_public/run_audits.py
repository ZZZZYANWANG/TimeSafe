"""Stage 2: run isolated TimeSafe audits and assemble their evidence.

Every model audit executes in a fresh Python interpreter.  This preserves the
real ACTR public learner while avoiding process-local native thread-pool state
from one histogram-boosting workload affecting a later audit.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from audit_plan import AUDIT_NAMES, EXPECTED


def worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[variable] = "1"
    environment.setdefault("PYTHONHASHSEED", "0")
    return environment


def run_worker(
    *,
    audit: str,
    actr_root: str | Path,
    output_dir: str | Path,
    environment: dict[str, str] | None = None,
) -> Path:
    if audit not in AUDIT_NAMES:
        raise ValueError(f"Unknown audit {audit!r}")
    here = Path(__file__).resolve().parent
    output = Path(output_dir).resolve()
    worker_dir = output / "audit_workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    target = worker_dir / f"{audit}.json"
    log_path = worker_dir / f"{audit}.log"
    command = [
        sys.executable,
        "-u",
        str(here / "audit_worker.py"),
        "--audit",
        audit,
        "--actr-root",
        str(Path(actr_root).resolve()),
        "--output",
        str(target),
    ]
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment or worker_environment(),
        )
    if completed.returncode:
        tail = log_path.read_text(encoding="utf-8")[-4000:]
        raise RuntimeError(f"ACTR audit worker {audit!r} failed:\n{tail}")
    return target


def combine_worker_artifacts(output_dir: str | Path) -> dict:
    output = Path(output_dir).resolve()
    worker_dir = output / "audit_workers"
    worker_payloads: dict[str, dict] = {}
    for audit in AUDIT_NAMES:
        target = worker_dir / f"{audit}.json"
        if not target.exists():
            raise FileNotFoundError(f"Missing audit worker artifact: {target}")
        payload = json.loads(target.read_text(encoding="utf-8"))
        if payload.get("audit") != audit or "result" not in payload:
            raise RuntimeError(f"Malformed worker artifact for {audit}: {target}")
        worker_payloads[audit] = payload

    observed = {name: payload["result"]["status"] for name, payload in worker_payloads.items()}
    unexpected = {
        name: status
        for name, status in observed.items()
        if status not in EXPECTED[name]
    }
    first = worker_payloads[AUDIT_NAMES[0]]
    payload = {
        "experiment": "ACTR-20 privacy-hardened public benchmark",
        "orchestration": "one_fresh_process_per_audit",
        "audit_asset_count": first["audit_asset_count"],
        "audit_rows": first["audit_rows"],
        "checkpoint_counts": {
            name: len(item.get("checkpoints", []))
            for name, item in worker_payloads.items()
        },
        **{name: item["result"] for name, item in worker_payloads.items()},
        "expected_verdicts": {name: sorted(values) for name, values in EXPECTED.items()},
        "observed_verdicts": observed,
        "all_expected": not unexpected,
        "worker_artifacts": {
            name: str((worker_dir / f"{name}.json").relative_to(output))
            for name in AUDIT_NAMES
        },
    }
    (output / "timesafe_audits.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if unexpected:
        raise RuntimeError(f"Unexpected ACTR TimeSafe verdicts: {unexpected}")
    return payload


def run(actr_root: str | Path, output_dir: str | Path) -> dict:
    environment = worker_environment()
    for position, audit in enumerate(AUDIT_NAMES, start=1):
        print(f"  [audit {position}/{len(AUDIT_NAMES)}] {audit}", flush=True)
        run_worker(
            audit=audit,
            actr_root=actr_root,
            output_dir=output_dir,
            environment=environment,
        )
    return combine_worker_artifacts(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actr-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--combine-only", action="store_true")
    args = parser.parse_args()
    if args.combine_only:
        result = combine_worker_artifacts(args.output_dir)
    else:
        if not args.actr_root:
            parser.error("--actr-root is required unless --combine-only is used")
        result = run(args.actr_root, args.output_dir)
    print(json.dumps(result["observed_verdicts"], indent=2))


if __name__ == "__main__":
    main()
