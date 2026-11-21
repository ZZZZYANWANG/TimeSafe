from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import quantum_open_system_factor_search as q
import quantum_open_system_factor_search_fast as f
from quantum_open_system_factor_search_run import corrected_fit_predict_pair

EPS = q.EPS
WINDOWS = (63, 126, 252)
BACKFLOW_PAIRS = ((10, 40), (20, 80))
NEW_OUT = Path(__file__).resolve().parent / "quantum_channel_memory_results"
NEW_OUT.mkdir(parents=True, exist_ok=True)
q.OUT = NEW_OUT
f.OUT = NEW_OUT
q.fit_predict_pair = corrected_fit_predict_pair
f.fit_predict_pair = corrected_fit_predict_pair


def _safe_entropy(norm: pd.Series) -> pd.Series:
    radius = norm.clip(0.0, 0.999999)
    p = ((1.0 + radius) / 2.0).clip(EPS, 1.0 - EPS)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def _strict_ar_innovation(series: pd.Series, window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    lag = series.shift(1)
    cov = series.rolling(window, min_periods=max(30, window // 2)).cov(lag).shift(1)
    var = lag.rolling(window, min_periods=max(30, window // 2)).var(ddof=0).shift(1)
    phi = (cov / (var + EPS)).clip(-1.5, 1.5)
    prediction = phi * lag
    innovation = series - prediction
    state_var = series.shift(1).rolling(window, min_periods=max(30, window // 2)).var(ddof=0)
    innovation_var = innovation.shift(1).rolling(window, min_periods=max(30, window // 2)).var(ddof=0)
    gain = (state_var / (state_var + innovation_var + EPS)).clip(0.0, 1.0)
    return prediction, innovation, gain


def _strict_two_regression(
    x: pd.Series,
    z: pd.Series,
    y: pd.Series,
    window: int,
    ridge: float = 0.05,
) -> tuple[pd.Series, pd.Series]:
    minp = max(30, window // 2)
    sxx = (x * x).shift(1).rolling(window, min_periods=minp).mean()
    sxz = (x * z).shift(1).rolling(window, min_periods=minp).mean()
    szz = (z * z).shift(1).rolling(window, min_periods=minp).mean()
    sxy = (x * y).shift(1).rolling(window, min_periods=minp).mean()
    szy = (z * y).shift(1).rolling(window, min_periods=minp).mean()
    scale = (sxx + szz).abs().fillna(0.0)
    lam = ridge * (scale + 1.0)
    a = sxx + lam
    d = szz + lam
    det = a * d - sxz * sxz
    beta_x = (sxy * d - szy * sxz) / (det + EPS)
    beta_z = (szy * a - sxy * sxz) / (det + EPS)
    return beta_x.clip(-2.0, 2.0), beta_z.clip(-2.0, 2.0)


def _memory_force(innovation: pd.Series, window: int, lags: int = 5) -> pd.Series:
    minp = max(30, window // 2)
    force = pd.Series(0.0, index=innovation.index)
    for lag in range(1, lags + 1):
        lagged = innovation.shift(lag)
        cov = innovation.rolling(window, min_periods=minp).cov(lagged).shift(1)
        var = lagged.rolling(window, min_periods=minp).var(ddof=0).shift(1)
        coefficient = (cov / (var + EPS)).clip(-0.75, 0.75)
        force = force + coefficient * lagged
    return force


def add_channel_memory_candidates(
    frame: pd.DataFrame,
    parent_cols: list[str],
) -> tuple[pd.DataFrame, list[q.CandidateMeta], list[str]]:
    pieces: list[pd.DataFrame] = []
    metas: dict[str, q.CandidateMeta] = {}
    controls: set[str] = set()

    for _, g0 in frame.groupby("ticker", sort=False):
        g = g0.sort_values("date").copy()
        bx = np.tanh(g["body_z"].astype(float) / 2.0)
        by = np.tanh(g["gap_z"].astype(float) / 2.0)
        bz = g["close_location"].astype(float).fillna(0.0).clip(-0.999, 0.999)
        norm = np.sqrt(bx * bx + by * by + bz * bz)
        scale = np.maximum(1.0, norm / 0.999)
        states = [bx / scale, by / scale, bz / scale]
        measurement = np.tanh(
            0.45 * g["range_z"].clip(lower=0).fillna(0.0)
            + 0.25 * g["volume_z"].clip(lower=0).fillna(0.0)
        ).clip(0.0, 1.0)
        trend = np.tanh(g["mom20"].astype(float) / 2.0)

        fast_states: dict[int, list[pd.Series]] = {}
        for span in {x for pair in BACKFLOW_PAIRS for x in pair}:
            fast_states[span] = [
                s.ewm(span=span, min_periods=max(6, span // 2), adjust=False).mean()
                for s in states
            ]

        for window in WINDOWS:
            predictions = []
            innovations = []
            gains = []
            for state in states:
                prediction, innovation, gain = _strict_ar_innovation(state, window)
                predictions.append(prediction)
                innovations.append(innovation)
                gains.append(gain)

            innovation_return = (
                gains[0] * innovations[0]
                + gains[1] * innovations[1]
                + 0.5 * gains[2] * innovations[2]
            )
            predicted_norm = np.sqrt(sum(x * x for x in predictions)).clip(0.0, 0.999999)
            observed_norm = np.sqrt(sum(x * x for x in states)).clip(0.0, 0.999999)
            entropy_production = _safe_entropy(observed_norm) - _safe_entropy(predicted_norm)

            persistence_responses = []
            nonunital_drifts = []
            for state in states:
                x = state.shift(1)
                z = x * measurement.shift(1)
                beta_x, beta_z = _strict_two_regression(x, z, state, window)
                persistence_responses.append((beta_x + beta_z).abs() - beta_x.abs())
                residual = state - beta_x * x - beta_z * z
                nonunital_drifts.append(
                    residual.shift(1).rolling(window, min_periods=max(30, window // 2)).mean()
                )
            persistence_response = sum(persistence_responses) / len(persistence_responses)
            nonunital_drift = (
                nonunital_drifts[0] + nonunital_drifts[1] + 0.5 * nonunital_drifts[2]
            )

            memory_force = _memory_force(innovation_return, window)
            innovation_std = innovation_return.shift(1).rolling(
                window, min_periods=max(30, window // 2)
            ).std(ddof=0)
            jump = (innovation_return / (innovation_std + EPS)).clip(-8.0, 8.0)

            dr2 = sum((state - state.shift(1)) ** 2 for state in states)
            dm2 = (measurement - measurement.shift(1)) ** 2
            qfi_proxy = np.log1p(dr2 / (dm2 + 0.025))
            qfi_surprise = q.strict_zscore(
                qfi_proxy, max(126, window), min_periods=max(50, window // 2)
            ).clip(-6.0, 6.0)

            raw_controls = {
                f"filter_innovation_return_w{window}": innovation_return,
                f"liouvillian_persistence_response_w{window}": persistence_response,
                f"channel_nonunital_drift_w{window}": nonunital_drift,
                f"memory_kernel_force_w{window}": memory_force,
                f"fisher_susceptibility_w{window}": qfi_surprise,
                f"entropy_production_w{window}": entropy_production,
                f"innovation_jump_w{window}": jump,
            }
            for name, values in raw_controls.items():
                g[name] = values.replace([np.inf, -np.inf], np.nan).clip(-20.0, 20.0)
                controls.add(name)

            definitions = {
                f"quantum_filter_measurement_innovation_w{window}": (
                    innovation_return * (0.20 + measurement)
                ),
                f"liouvillian_zeno_response_w{window}": (
                    trend * persistence_response * (0.20 + measurement)
                ),
                f"channel_nonunital_pointer_drift_w{window}": (
                    nonunital_drift * (0.20 + measurement)
                ),
                f"nakajima_zwanzig_memory_force_w{window}": memory_force,
                f"memory_release_trend_switch_w{window}": (
                    -trend * memory_force.abs() * (0.20 + measurement)
                ),
                f"fisher_measurement_susceptibility_response_w{window}": (
                    innovation_return * qfi_surprise * (0.20 + measurement)
                ),
                f"channel_entropy_production_response_w{window}": (
                    -trend * entropy_production * (0.20 + measurement)
                ),
                f"quantum_jump_backaction_w{window}": (
                    -jump * jump.abs() * (0.20 + measurement)
                ),
            }
            mechanism_map = {
                "quantum_filter_measurement_innovation": "quantum_filter_innovation",
                "liouvillian_zeno_response": "liouvillian_zeno_anti_zeno",
                "channel_nonunital_pointer_drift": "cptp_channel_nonunitality",
                "nakajima_zwanzig_memory_force": "projection_operator_memory",
                "memory_release_trend_switch": "projection_operator_memory",
                "fisher_measurement_susceptibility_response": "quantum_fisher_susceptibility",
                "channel_entropy_production_response": "open_system_entropy_production",
                "quantum_jump_backaction": "jump_diffusion_backaction",
            }
            for name, raw in definitions.items():
                standardized = q.strict_zscore(
                    pd.Series(raw, index=g.index), 252, min_periods=80
                ).clip(-8.0, 8.0)
                g[name] = standardized
                root = name.rsplit("_w", 1)[0]
                metas[name] = q.CandidateMeta(
                    candidate=name,
                    mechanism=mechanism_map[root],
                    span=window,
                    formula_role=root,
                )

        for fast, slow in BACKFLOW_PAIRS:
            difference = [
                fast_states[fast][i] - fast_states[slow][i]
                for i in range(3)
            ]
            trace_distance = 0.5 * np.sqrt(sum(x * x for x in difference))
            backflow = (trace_distance - trace_distance.shift(1)).clip(lower=0.0)
            return_projection = difference[0] + difference[1] + 0.5 * difference[2]
            control_name = f"trace_distance_backflow_{fast}_{slow}"
            g[control_name] = backflow
            controls.add(control_name)
            definitions = {
                f"nonmarkov_information_backflow_direction_{fast}_{slow}": (
                    return_projection * backflow * (0.20 + measurement)
                ),
                f"nonmarkov_backflow_trend_switch_{fast}_{slow}": (
                    -trend * np.sign(return_projection) * backflow * (0.20 + measurement)
                ),
            }
            for name, raw in definitions.items():
                g[name] = q.strict_zscore(
                    pd.Series(raw, index=g.index), 252, min_periods=80
                ).clip(-8.0, 8.0)
                root = name.rsplit("_", 2)[0]
                mechanism = (
                    "nonmarkov_information_backflow"
                    if name.startswith("nonmarkov_information")
                    else "nonmarkov_backflow_switch"
                )
                metas[name] = q.CandidateMeta(
                    candidate=name,
                    mechanism=mechanism,
                    span=slow,
                    formula_role=root,
                )
        pieces.append(g)

    out = pd.concat(pieces, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    enriched_parents = list(dict.fromkeys(parent_cols + sorted(controls)))
    return out, list(metas.values()), enriched_parents


def write_preregistration(candidates: list[q.CandidateMeta], parents: list[str]) -> None:
    payload = {
        "schema_version": 1,
        "preregistered_before_outer_results_were_read": True,
        "theory_families": sorted({x.mechanism for x in candidates}),
        "candidate_identities": [x.candidate for x in candidates],
        "candidate_count_per_horizon": len(candidates),
        "horizons": list(q.HORIZONS),
        "declarations": len(candidates) * len(q.HORIZONS),
        "raw_component_controls_in_parent": [
            x for x in parents
            if any(token in x for token in (
                "filter_innovation", "liouvillian", "nonunital", "memory_kernel",
                "fisher_susceptibility", "entropy_production", "innovation_jump",
                "trace_distance_backflow",
            ))
        ],
        "selection": {
            "orientation": "DEV_A only",
            "replication": ["DEV_B1", "DEV_B2"],
            "outer": ["VAL", "HIST", "LATE", "FRESH"],
            "protocol": "same strict fast protocol as round one",
        },
        "production_boundary": "continuous public proxies only; any passing identity remains default-off pending private roll-safe replay",
    }
    (NEW_OUT / "PREREGISTRATION.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    futures_panel, futures_coverage = q.download_panel()
    base_frame, _, base_parents = q.build_features(futures_panel)
    futures_frame, candidates, parent_cols = add_channel_memory_candidates(base_frame, base_parents)
    write_preregistration(candidates, parent_cols)
    for h in q.HORIZONS:
        invalid = futures_frame[f"target_raw_h{h}"].abs() > 0.35 * math.sqrt(h)
        futures_frame.loc[
            invalid,
            [f"target_raw_h{h}", f"target_norm_h{h}", f"target_open_h{h}"],
        ] = np.nan

    screen = f.fast_screen(futures_frame, candidates, parent_cols)
    finalists = f.fast_select(screen)

    stock_panel, stock_coverage = f.download_stock_panel()
    if stock_panel.empty:
        stock_frame = pd.DataFrame()
        stock_parents = []
    else:
        stock_base, _, stock_base_parents = q.build_features(stock_panel)
        stock_frame, _, stock_parents = add_channel_memory_candidates(
            stock_base, stock_base_parents
        )
        for h in q.HORIZONS:
            invalid = stock_frame[f"target_raw_h{h}"].abs() > 0.35 * math.sqrt(h)
            stock_frame.loc[
                invalid,
                [f"target_raw_h{h}", f"target_norm_h{h}", f"target_open_h{h}"],
            ] = np.nan

    results = []
    for _, row in finalists.iterrows():
        item = q.evaluate_finalist(futures_frame, screen, row, parent_cols)
        if stock_frame.empty or row["candidate"] not in stock_frame:
            transfer = {
                "universe": "unavailable",
                "direct": [],
                "ridge_normalized_increment": [],
                "val_hist_positive": False,
            }
        else:
            transfer = f.stock_transfer(
                stock_frame,
                stock_parents,
                str(row["candidate"]),
                float(row["orientation"]),
                int(row["horizon"]),
            )
        item["stock_domain_transfer"] = transfer
        item["historical_default_off_factor"] = bool(
            item["historical_default_off_factor"] and transfer["val_hist_positive"]
        )
        item["production_independent_alpha"] = False
        item["gates"]["stock_val_hist_direct_and_ridge_positive"] = bool(
            transfer["val_hist_positive"]
        )
        item["gates"]["canonical_roll_safe_execution_replay"] = False
        results.append(item)

    admitted = [x for x in results if x["historical_default_off_factor"]]
    payload = {
        "schema_version": 1,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "study": "quantum_channel_memory_preregistered_round",
        "data": {
            "futures_assets": int(futures_coverage["accepted"].sum()),
            "futures_rows": int(len(futures_frame)),
            "stock_assets": int(stock_coverage["accepted"].sum()) if len(stock_coverage) else 0,
            "stock_rows": int(len(stock_frame)),
            "fresh_cutoff": "2026-09-11",
        },
        "candidate_family_size": len(candidates) * len(q.HORIZONS),
        "parent_feature_count": len(parent_cols),
        "finalists": results,
        "admitted_historical_default_off": [x["candidate"] for x in admitted],
        "admitted_production_independent_alpha": [],
        "hourly_homologue": {"status": "deferred_until_private_replay"},
    }
    (NEW_OUT / "RESULTS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    f.write_report(payload)
    print(json.dumps({
        "candidate_declarations": payload["candidate_family_size"],
        "finalists": len(results),
        "historical_default_off_admissions": payload["admitted_historical_default_off"],
        "production_admissions": [],
    }, indent=2))


if __name__ == "__main__":
    main()
