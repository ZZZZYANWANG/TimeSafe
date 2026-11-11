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
from common import (
    ARTIFACT_SCHEMA_VERSION,
    AUDIT_CONTRACT_VERSION,
    EXPECTED_BENCHMARK_SHA256,
    EXPECTED_MANIFEST_SHA256,
    atomic_write_json,
    file_sha256,
)
from timesafe_quant import __version__ as timesafe_version


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
    run_id: str,
) -> Path:
    if audit not in AUDIT_NAMES:
        raise ValueError(f"Unknown audit {audit!r}")
    here = Path(__file__).resolve().parent
    output = Path(output_dir).resolve()
    worker_dir = output / "audit_workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    target = worker_dir / f"{audit}.json"
    target.unlink(missing_ok=True)
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
        "--run-id",
        run_id,
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


def _validate_shared_identity(
    worker_payloads: dict[str, dict],
    benchmark_config: dict,
) -> dict:
    """Reject stale or mixed worker artifacts before aggregating verdicts."""

    required = {
        "artifact_schema_version",
        "experiment_run_id",
        "implementation_sha256",
        "audit_contract_version",
        "benchmark_sha256",
        "actr_manifest_sha256",
        "benchmark_rows",
        "benchmark_sessions",
        "timesafe_version",
        "audit_asset_ids_sha256",
        "audit_asset_count",
        "audit_rows",
        "audit_sessions",
        "audit_session_min",
        "audit_session_max",
        "planned_checkpoints",
    }
    first_name = AUDIT_NAMES[0]
    identity = worker_payloads[first_name].get("identity")
    if not isinstance(identity, dict):
        raise RuntimeError(f"Worker {first_name!r} is missing an identity block.")
    missing = required.difference(identity)
    if missing:
        raise RuntimeError(f"Worker identity is missing fields: {sorted(missing)}")

    for name, worker in worker_payloads.items():
        candidate = worker.get("identity")
        if not isinstance(candidate, dict):
            raise RuntimeError(f"Worker {name!r} is missing a valid identity block.")
        if candidate != identity:
            differing = sorted(
                key
                for key in required.union(candidate or {})
                if (candidate or {}).get(key) != identity.get(key)
            )
            raise RuntimeError(
                f"Mixed or stale worker identity for {name!r}; differing fields: {differing}."
            )
        if worker.get("audit_asset_count") != identity["audit_asset_count"]:
            raise RuntimeError(f"Worker {name!r} asset count disagrees with its identity.")
        if worker.get("audit_rows") != identity["audit_rows"]:
            raise RuntimeError(f"Worker {name!r} row count disagrees with its identity.")

    expected = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "implementation_sha256": benchmark_config.get("implementation_sha256"),
        "audit_contract_version": AUDIT_CONTRACT_VERSION,
        "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
        "actr_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "timesafe_version": timesafe_version,
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise RuntimeError(
                f"Audit identity {key!r} mismatch: observed={identity.get(key)!r}, "
                f"expected={value!r}."
            )

    benchmark_expected = {
        "experiment_run_id": identity["experiment_run_id"],
        "implementation_sha256": identity["implementation_sha256"],
        "experiment_contract_version": identity["audit_contract_version"],
        "timesafe_version": identity["timesafe_version"],
        "data_sha256": identity["benchmark_sha256"],
        "actr_manifest_sha256": identity["actr_manifest_sha256"],
        "rows": identity["benchmark_rows"],
        "sessions": identity["benchmark_sessions"],
    }
    for key, value in benchmark_expected.items():
        if benchmark_config.get(key) != value:
            raise RuntimeError(
                f"Benchmark config and worker identity disagree on {key!r}: "
                f"config={benchmark_config.get(key)!r}, workers={value!r}."
            )
    return identity


def combine_worker_artifacts(output_dir: str | Path) -> dict:
    output = Path(output_dir).resolve()
    worker_dir = output / "audit_workers"
    config_path = output / "benchmark_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            "benchmark_config.json is required so worker artifacts cannot be "
            "combined against a stale or different ACTR benchmark."
        )
    benchmark_config = json.loads(config_path.read_text(encoding="utf-8"))
    worker_payloads: dict[str, dict] = {}
    worker_hashes: dict[str, str] = {}
    for audit in AUDIT_NAMES:
        target = worker_dir / f"{audit}.json"
        if not target.exists():
            raise FileNotFoundError(f"Missing audit worker artifact: {target}")
        payload = json.loads(target.read_text(encoding="utf-8"))
        if payload.get("audit") != audit or "result" not in payload:
            raise RuntimeError(f"Malformed worker artifact for {audit}: {target}")
        worker_payloads[audit] = payload
        worker_hashes[audit] = file_sha256(target)

    identity = _validate_shared_identity(worker_payloads, benchmark_config)
    observed = {name: payload["result"]["status"] for name, payload in worker_payloads.items()}
    unexpected = {
        name: status
        for name, status in observed.items()
        if status not in EXPECTED[name]
    }
    payload = {
        "experiment": "ACTR-20 privacy-hardened public benchmark",
        "orchestration": "one_fresh_process_per_audit",
        "identity": identity,
        "audit_asset_count": identity["audit_asset_count"],
        "audit_rows": identity["audit_rows"],
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
        "worker_artifact_sha256": worker_hashes,
    }
    atomic_write_json(output / "timesafe_audits.json", payload)
    if unexpected:
        raise RuntimeError(f"Unexpected ACTR TimeSafe verdicts: {unexpected}")
    return payload


def run(actr_root: str | Path, output_dir: str | Path) -> dict:
    output = Path(output_dir).resolve()
    config_path = output / "benchmark_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            "Run the benchmark stage first so the audit run identity is available."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_id = config.get("experiment_run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("benchmark_config.json is missing experiment_run_id.")
    environment = worker_environment()
    for position, audit in enumerate(AUDIT_NAMES, start=1):
        print(f"  [audit {position}/{len(AUDIT_NAMES)}] {audit}", flush=True)
        run_worker(
            audit=audit,
            actr_root=actr_root,
            output_dir=output_dir,
            environment=environment,
            run_id=run_id,
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
