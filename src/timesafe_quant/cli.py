"""Command line interface for TimeSafe."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable

import pandas as pd

from .availability import AvailabilityRule, audit_availability_invariance
from .demo import run_demo_report
from .maturity import LabelContract, audit_label_maturity, build_label_intervals
from .prefix import audit_determinism, audit_prefix_invariance
from .report import make_report, render_markdown, write_report
from .types import AuditStatus


def _load_callable(spec: str) -> Callable:
    if ":" not in spec:
        raise ValueError("Callable must use 'module:function' syntax.")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError(f"{spec!r} is not callable.")
    return function


def _read_panel(path: str, time_col: str, asset_col: str | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if time_col not in frame.columns:
        raise ValueError(f"CSV is missing time column {time_col!r}.")
    frame[time_col] = pd.to_datetime(frame[time_col], errors="raise")
    if asset_col:
        if asset_col not in frame.columns:
            raise ValueError(f"CSV is missing asset column {asset_col!r}.")
        frame = frame.set_index([time_col, asset_col])
    else:
        frame = frame.set_index(time_col)
    return frame


def _print_or_write(report, output: str | None) -> None:
    if output:
        path = write_report(report, output)
        print(path)
    else:
        print(render_markdown(report))


def _exit_code(status: AuditStatus) -> int:
    if status is AuditStatus.PASS:
        return 0
    if status is AuditStatus.WARN:
        return 1
    if status in {AuditStatus.FAIL, AuditStatus.INCONCLUSIVE}:
        return 2
    return 3


def _parse_availability_rule(spec: str) -> AvailabilityRule:
    parts = spec.split(":")
    if len(parts) not in {2, 3, 4}:
        raise ValueError(
            "Availability rule must use "
            "value_column:available_at_column[:event[:reference]]."
        )
    value_column, available_at_column = parts[:2]
    event = parts[2] if len(parts) >= 3 else "after_close"
    reference = parts[3] if len(parts) == 4 else "checkpoint"
    return AvailabilityRule(
        value_column=value_column,
        available_at_column=available_at_column,
        available_event=event,
        reference=reference,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="timesafe",
        description="Temporal integrity auditing for quantitative research.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run deterministic leakage and maturity examples.")
    demo.add_argument("--output", help="Write .md or .json instead of printing Markdown.")

    prefix = subparsers.add_parser("prefix", help="Audit one pandas transform for look-ahead leakage.")
    prefix.add_argument("--input", required=True, help="CSV input file.")
    prefix.add_argument("--callable", required=True, dest="callable_spec", help="module:function")
    prefix.add_argument("--time-col", required=True)
    prefix.add_argument("--asset-col")
    prefix.add_argument(
        "--checkpoint-policy",
        choices=["uniform", "all"],
        default="uniform",
        help="Use 'all' for a conclusive PASS; clean sampled audits return WARN.",
    )
    prefix.add_argument("--checkpoints", type=int, default=12)
    prefix.add_argument("--min-history", type=int, default=30)
    prefix.add_argument("--atol", type=float, default=1e-10)
    prefix.add_argument("--rtol", type=float, default=1e-8)
    prefix.add_argument("--determinism-runs", type=int, default=3)
    prefix.add_argument("--output")

    availability = subparsers.add_parser(
        "availability",
        help="Audit dependence on values materialized before declared availability.",
    )
    availability.add_argument("--input", required=True, help="CSV input file.")
    availability.add_argument("--callable", required=True, dest="callable_spec", help="module:function")
    availability.add_argument("--time-col", required=True)
    availability.add_argument("--asset-col")
    availability.add_argument(
        "--rule",
        action="append",
        required=True,
        help=(
            "value_column:available_at_column[:event[:reference]]; "
            "reference is checkpoint (labels) or row (historical features)."
        ),
    )
    availability.add_argument("--decision-event", default="close")
    availability.add_argument(
        "--checkpoint-policy", choices=["uniform", "all"], default="uniform"
    )
    availability.add_argument("--checkpoints", type=int, default=12)
    availability.add_argument("--min-history", type=int, default=30)
    availability.add_argument("--atol", type=float, default=1e-10)
    availability.add_argument("--rtol", type=float, default=1e-8)
    availability.add_argument("--output")

    maturity = subparsers.add_parser("maturity", help="Derive label maturity and purge requirements.")
    maturity.add_argument("--sessions", required=True, help="CSV containing a session column.")
    maturity.add_argument("--time-col", required=True)
    maturity.add_argument("--execution-offset", type=int, default=1)
    maturity.add_argument("--holding-period", type=int, default=20)
    maturity.add_argument("--publication-lag", type=int, default=0)
    maturity.add_argument("--safety-gap", type=int, default=0)
    maturity.add_argument("--decision-event", default="close")
    maturity.add_argument("--execution-event", default="open")
    maturity.add_argument("--exit-event", default="close")
    maturity.add_argument("--label-available-event", default="after_close")
    maturity.add_argument("--as-of")
    maturity.add_argument("--as-of-event", default="after_close")
    maturity.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "demo":
            report = run_demo_report()
            _print_or_write(report, args.output)
            return 0

        if args.command == "prefix":
            panel = _read_panel(args.input, args.time_col, args.asset_col)
            function = _load_callable(args.callable_spec)
            prefix_result = audit_prefix_invariance(
                function,
                panel,
                time_level=args.time_col,
                checkpoint_policy=args.checkpoint_policy,
                n_checkpoints=args.checkpoints,
                min_history=args.min_history,
                atol=args.atol,
                rtol=args.rtol,
            )
            determinism = audit_determinism(
                function,
                panel,
                repetitions=args.determinism_runs,
                atol=args.atol,
                rtol=args.rtol,
            )
            report = make_report(prefix=prefix_result, determinism=determinism)
            _print_or_write(report, args.output)
            return _exit_code(report.overall_status)

        if args.command == "availability":
            panel = _read_panel(args.input, args.time_col, args.asset_col)
            function = _load_callable(args.callable_spec)
            rules = [_parse_availability_rule(spec) for spec in args.rule]
            for rule in rules:
                if rule.available_at_column in panel.columns:
                    panel[rule.available_at_column] = pd.to_datetime(
                        panel[rule.available_at_column], errors="raise"
                    )
            result = audit_availability_invariance(
                function,
                panel,
                rules=rules,
                time_level=args.time_col,
                checkpoint_policy=args.checkpoint_policy,
                n_checkpoints=args.checkpoints,
                min_history=args.min_history,
                decision_event=args.decision_event,
                atol=args.atol,
                rtol=args.rtol,
            )
            report = make_report(prefix=result)
            _print_or_write(report, args.output)
            return _exit_code(report.overall_status)

        if args.command == "maturity":
            frame = pd.read_csv(args.sessions)
            if args.time_col not in frame.columns:
                raise ValueError(f"CSV is missing time column {args.time_col!r}.")
            sessions = pd.DatetimeIndex(
                pd.to_datetime(frame[args.time_col], errors="raise")
            ).unique().sort_values()
            contract = LabelContract(
                execution_offset_sessions=args.execution_offset,
                holding_period_sessions=args.holding_period,
                label_publication_lag_sessions=args.publication_lag,
                safety_gap_sessions=args.safety_gap,
                decision_event=args.decision_event,
                execution_event=args.execution_event,
                exit_event=args.exit_event,
                label_available_event=args.label_available_event,
            )
            intervals = build_label_intervals(
                sessions, calendar=sessions, contract=contract
            )
            as_of = pd.Timestamp(args.as_of) if args.as_of else sessions[-1]
            result = audit_label_maturity(
                intervals,
                contract=contract,
                as_of=as_of,
                as_of_event=args.as_of_event,
                calendar=sessions,
            )
            report = make_report(maturity=result)
            _print_or_write(report, args.output)
            return _exit_code(report.overall_status)

        parser.error(f"Unknown command: {args.command}")
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}),
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
