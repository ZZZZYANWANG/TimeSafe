from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import quantum_open_system_factor_search as q

warnings.filterwarnings("ignore")

OUT = q.OUT
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(q.SEED + 17)
EPS = q.EPS
MAX_FINALISTS = 4
HGB_TRAIN_CAP = 60000

STOCKS = {
    "AAPL": "technology",
    "MSFT": "technology",
    "NVDA": "technology",
    "AVGO": "technology",
    "AMD": "technology",
    "ORCL": "technology",
    "AMZN": "consumer",
    "META": "communication",
    "GOOGL": "communication",
    "TSLA": "consumer",
    "JPM": "financial",
    "BAC": "financial",
    "GS": "financial",
    "MS": "financial",
    "XOM": "energy",
    "CVX": "energy",
    "CAT": "industrial",
    "GE": "industrial",
    "BA": "industrial",
    "WMT": "consumer_staples",
    "COST": "consumer_staples",
    "KO": "consumer_staples",
    "PEP": "consumer_staples",
    "UNH": "healthcare",
    "JNJ": "healthcare",
    "LLY": "healthcare",
    "PFE": "healthcare",
    "HD": "consumer",
    "MCD": "consumer",
    "NKE": "consumer",
}


def model_pipeline(model_type: str, numeric_cols: list[str], categorical_cols: list[str]):
    if model_type == "ridge":
        numeric = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers = [("num", numeric, numeric_cols)]
        if categorical_cols:
            transformers.append((
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=5),
                categorical_cols,
            ))
        pre = ColumnTransformer(transformers)
        model = Ridge(alpha=25.0, solver="lsqr", fit_intercept=True)
    elif model_type == "hgb":
        pre = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]), numeric_cols),
        ])
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=70,
            max_leaf_nodes=15,
            min_samples_leaf=100,
            l2_regularization=1.0,
            random_state=q.SEED,
        )
    else:
        raise ValueError(model_type)
    return Pipeline([("pre", pre), ("model", model)])


q._model_pipeline = model_pipeline


def fit_predict_pair(
    train: pd.DataFrame,
    test: pd.DataFrame,
    parent_cols: list[str],
    candidate: str,
    target: str,
    model_type: str,
) -> pd.DataFrame:
    categorical = ["ticker", "sector"] if model_type == "ridge" else []
    tr_cols = parent_cols + categorical + [candidate, target]
    te_cols = parent_cols + categorical + [candidate, target, "date", "ticker"]
    tr = train[tr_cols].copy()
    te = test[te_cols].copy()
    tr = tr[
        np.isfinite(pd.to_numeric(tr[target], errors="coerce"))
        & np.isfinite(pd.to_numeric(tr[candidate], errors="coerce"))
    ]
    te = te[
        np.isfinite(pd.to_numeric(te[target], errors="coerce"))
        & np.isfinite(pd.to_numeric(te[candidate], errors="coerce"))
    ]
    if len(tr) < 2500 or len(te) < 80:
        return pd.DataFrame()
    if model_type == "hgb" and len(tr) > HGB_TRAIN_CAP:
        tr = tr.sample(HGB_TRAIN_CAP, random_state=q.SEED).sort_index()
    parent_pipe = model_pipeline(model_type, parent_cols, categorical)
    joint_pipe = model_pipeline(model_type, parent_cols + [candidate], categorical)
    parent_pipe.fit(tr[parent_cols + categorical], tr[target].astype(float))
    joint_pipe.fit(tr[parent_cols + [candidate] + categorical], tr[target].astype(float))
    out = te[["date", "ticker", target]].copy()
    out["parent_prediction"] = parent_pipe.predict(te[parent_cols + categorical])
    out["joint_prediction"] = joint_pipe.predict(te[parent_cols + [candidate] + categorical])
    out["parent_loss"] = (out[target] - out["parent_prediction"]) ** 2
    out["joint_loss"] = (out[target] - out["joint_prediction"]) ** 2
    return out


q.fit_predict_pair = fit_predict_pair


def _mean_product_ic(frame: pd.DataFrame, signal: str, target: str, min_n: int = 35) -> float:
    values = []
    for _, g in frame.groupby("ticker"):
        value = q.safe_spearman(g[signal], g[target], min_n=min_n)
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else np.nan


def _mean_daily_ic(frame: pd.DataFrame, signal: str, target: str, min_n: int = 8) -> float:
    values = []
    for _, g in frame.groupby("date"):
        value = q.safe_spearman(g[signal], g[target], min_n=min_n)
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else np.nan


def _conditional_scalar_gain(frame: pd.DataFrame, signal: str, residual: str) -> float:
    dates = sorted(frame["date"].dropna().unique())
    if len(dates) < 80:
        return np.nan
    split = dates[len(dates) // 2]
    fit = frame[frame["date"] < split][[signal, residual]].dropna()
    test = frame[frame["date"] >= split][[signal, residual]].dropna()
    if len(fit) < 500 or len(test) < 500:
        return np.nan
    x = fit[signal].to_numpy(float)
    y = fit[residual].to_numpy(float)
    beta = float(np.dot(x, y) / (np.dot(x, x) + 25.0))
    yt = test[residual].to_numpy(float)
    xt = test[signal].to_numpy(float)
    base = float(np.mean(yt * yt))
    joint = float(np.mean((yt - beta * xt) ** 2))
    return 100.0 * (base - joint) / (base + EPS)


def _parent_residual_block(
    frame: pd.DataFrame,
    parent_cols: list[str],
    horizon: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    target = f"target_norm_h{horizon}"
    train = frame[
        (frame[f"target_end_date_h{horizon}"] < start)
        & (frame["date"] >= pd.Timestamp("2008-01-01"))
    ].copy()
    test = frame[frame["date"].between(start, end)].copy()
    train = train[np.isfinite(pd.to_numeric(train[target], errors="coerce"))]
    test = test[np.isfinite(pd.to_numeric(test[target], errors="coerce"))]
    if len(train) < 2500 or len(test) < 500:
        return pd.DataFrame()
    pipe = model_pipeline("ridge", parent_cols, ["ticker", "sector"])
    pipe.fit(train[parent_cols + ["ticker", "sector"]], train[target].astype(float))
    out = test[["date", "ticker", target]].copy()
    out["parent_residual"] = out[target] - pipe.predict(test[parent_cols + ["ticker", "sector"]])
    return out[["date", "ticker", "parent_residual"]]


def fast_screen(
    frame: pd.DataFrame,
    candidates: list[q.CandidateMeta],
    parent_cols: list[str],
) -> pd.DataFrame:
    residual_cache: dict[tuple[int, str], pd.DataFrame] = {}
    for h in q.HORIZONS:
        residual_cache[(h, "DEV_B1")] = _parent_residual_block(
            frame, parent_cols, h, pd.Timestamp("2014-01-01"), pd.Timestamp("2015-12-31")
        )
        residual_cache[(h, "DEV_B2")] = _parent_residual_block(
            frame, parent_cols, h, pd.Timestamp("2016-01-01"), pd.Timestamp("2018-12-31")
        )

    rows = []
    for h in q.HORIZONS:
        target = f"target_norm_h{h}"
        for meta in candidates:
            dev_a = frame.loc[q.phase_mask(frame, "DEV_A"), ["date", "ticker", meta.candidate, target]].copy()
            raw_ic = _mean_product_ic(dev_a, meta.candidate, target)
            orientation = 1.0 if np.isfinite(raw_ic) and raw_ic >= 0 else -1.0
            temp = f"__oriented_{meta.candidate}"
            frame[temp] = orientation * frame[meta.candidate]
            direct = {}
            for phase in ("DEV_A", "DEV_B1", "DEV_B2"):
                sub = frame.loc[q.phase_mask(frame, phase), ["date", "ticker", temp, target]]
                direct[phase] = {
                    "tsic": _mean_product_ic(sub, temp, target),
                    "csic": _mean_daily_ic(sub, temp, target),
                }
            conditional = {}
            for phase in ("DEV_B1", "DEV_B2"):
                residual = residual_cache[(h, phase)]
                merged = frame[["date", "ticker", temp]].merge(
                    residual, on=["date", "ticker"], how="inner", validate="one_to_one"
                )
                conditional[phase] = {
                    "residual_tsic": _mean_product_ic(merged, temp, "parent_residual"),
                    "residual_csic": _mean_daily_ic(merged, temp, "parent_residual"),
                    "scalar_oos_gain_pct": _conditional_scalar_gain(merged, temp, "parent_residual"),
                }
            frame.drop(columns=[temp], inplace=True)
            rows.append({
                "candidate": meta.candidate,
                "mechanism": meta.mechanism,
                "span": meta.span,
                "formula_role": meta.formula_role,
                "horizon": h,
                "orientation": orientation,
                "dev_a_tsic": direct["DEV_A"]["tsic"],
                "dev_b1_tsic": direct["DEV_B1"]["tsic"],
                "dev_b2_tsic": direct["DEV_B2"]["tsic"],
                "dev_a_csic": direct["DEV_A"]["csic"],
                "dev_b1_csic": direct["DEV_B1"]["csic"],
                "dev_b2_csic": direct["DEV_B2"]["csic"],
                "dev_b1_residual_tsic": conditional["DEV_B1"]["residual_tsic"],
                "dev_b2_residual_tsic": conditional["DEV_B2"]["residual_tsic"],
                "dev_b1_residual_csic": conditional["DEV_B1"]["residual_csic"],
                "dev_b2_residual_csic": conditional["DEV_B2"]["residual_csic"],
                "dev_fold1_ridge_gain_pct": conditional["DEV_B1"]["scalar_oos_gain_pct"],
                "dev_fold2_ridge_gain_pct": conditional["DEV_B2"]["scalar_oos_gain_pct"],
            })
    out = pd.DataFrame(rows)
    out["min_dev_tsic"] = out[["dev_a_tsic", "dev_b1_tsic", "dev_b2_tsic"]].min(axis=1)
    out["min_dev_csic"] = out[["dev_a_csic", "dev_b1_csic", "dev_b2_csic"]].min(axis=1)
    out["min_residual_tsic"] = out[["dev_b1_residual_tsic", "dev_b2_residual_tsic"]].min(axis=1)
    out["min_residual_csic"] = out[["dev_b1_residual_csic", "dev_b2_residual_csic"]].min(axis=1)
    out["min_dev_gain"] = out[["dev_fold1_ridge_gain_pct", "dev_fold2_ridge_gain_pct"]].min(axis=1)
    out["dev_pass"] = (
        (out["min_dev_tsic"] > 0.0025)
        & (out["min_residual_tsic"] > 0.0)
        & (out["min_dev_gain"] > 0.0)
        & out[[
            "dev_a_tsic", "dev_b1_tsic", "dev_b2_tsic",
            "dev_b1_residual_tsic", "dev_b2_residual_tsic",
            "dev_fold1_ridge_gain_pct", "dev_fold2_ridge_gain_pct",
        ]].notna().all(axis=1)
    )
    out["selection_score"] = (
        out["min_dev_tsic"].clip(-0.1, 0.1)
        + 0.75 * out["min_residual_tsic"].clip(-0.1, 0.1)
        + 0.02 * out["min_dev_gain"].clip(-1.0, 1.0)
    )
    out.to_csv(OUT / "candidate_screen.csv", index=False)
    return out


def fast_select(screen: pd.DataFrame) -> pd.DataFrame:
    passed = screen[screen["dev_pass"]].copy()
    if passed.empty:
        passed = screen.sort_values("selection_score", ascending=False).head(MAX_FINALISTS * 3).copy()
        passed["selection_note"] = "diagnostic_only_no_full_development_pass"
    else:
        passed["selection_note"] = "passed_development_gate"
    passed = (
        passed.sort_values(
            ["horizon", "mechanism", "selection_score"],
            ascending=[True, True, False],
        )
        .groupby(["horizon", "mechanism"], as_index=False)
        .head(1)
    )
    selected_rows = []
    horizon_count: dict[int, int] = {}
    mechanism_count: dict[str, int] = {}
    for _, row in passed.sort_values("selection_score", ascending=False).iterrows():
        h = int(row["horizon"])
        mechanism = str(row["mechanism"])
        if horizon_count.get(h, 0) >= 2 or mechanism_count.get(mechanism, 0) >= 1:
            continue
        selected_rows.append(row)
        horizon_count[h] = horizon_count.get(h, 0) + 1
        mechanism_count[mechanism] = mechanism_count.get(mechanism, 0) + 1
        if len(selected_rows) >= MAX_FINALISTS:
            break
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(OUT / "selected_finalists.csv", index=False)
    return selected


def period_oof_predictions(
    frame: pd.DataFrame,
    parent_cols: list[str],
    candidate: str,
    horizon: int,
    target_kind: str,
    model_type: str,
) -> pd.DataFrame:
    target = f"target_{target_kind}_h{horizon}"
    pieces = []
    for phase in ("VAL", "HIST", "LATE", "FRESH"):
        start, end = q.PHASES[phase]
        train = frame[
            (frame[f"target_end_date_h{horizon}"] < start)
            & (frame["date"] >= pd.Timestamp("2008-01-01"))
        ]
        test = frame[frame["date"].between(start, end)]
        pred = fit_predict_pair(train, test, parent_cols, candidate, target, model_type)
        if pred.empty:
            continue
        pred["fit_cutoff"] = str(start.date())
        pred["phase_fit"] = phase
        pieces.append(pred)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


q.annual_oof_predictions = period_oof_predictions


def download_stock_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    coverage = []
    for ticker, sector in STOCKS.items():
        data = pd.DataFrame()
        error = None
        for attempt in range(3):
            try:
                data = yf.download(
                    ticker,
                    start=q.START,
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                    threads=False,
                    timeout=30,
                )
                data = q._flatten_columns(data)
                if len(data) >= 750:
                    break
            except Exception as exc:
                error = repr(exc)
            time.sleep(1.0 + attempt)
        needed = {"open", "high", "low", "close", "volume"}
        if data.empty or not needed.issubset(data.columns):
            coverage.append({"ticker": ticker, "accepted": False, "rows": len(data), "reason": error or "missing"})
            continue
        data = data.reset_index()
        date_col = "date" if "date" in data.columns else data.columns[0]
        data = data.rename(columns={date_col: "date"})
        data["date"] = pd.to_datetime(data["date"], utc=True).dt.tz_convert(None).dt.normalize()
        for col in needed:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna(subset=["date", "open", "high", "low", "close"])
        data = data[(data[["open", "high", "low", "close"]] > 0).all(axis=1)]
        data = data.sort_values("date").drop_duplicates("date", keep="last")
        accepted = len(data) >= 750
        coverage.append({
            "ticker": ticker,
            "accepted": accepted,
            "rows": len(data),
            "start": str(data["date"].min().date()) if len(data) else None,
            "end": str(data["date"].max().date()) if len(data) else None,
            "reason": None if accepted else "insufficient_history",
        })
        if accepted:
            data["ticker"] = ticker
            data["sector"] = sector
            frames.append(data[["date", "ticker", "sector", "open", "high", "low", "close", "volume"]])
    coverage_df = pd.DataFrame(coverage)
    coverage_df.to_csv(OUT / "stock_data_coverage.csv", index=False)
    if not frames:
        return pd.DataFrame(), coverage_df
    panel = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    return panel, coverage_df


def stock_transfer(
    stock_frame: pd.DataFrame,
    stock_parent_cols: list[str],
    candidate: str,
    orientation: float,
    horizon: int,
) -> dict:
    oriented = f"stock_oriented__{candidate}"
    stock_frame[oriented] = orientation * stock_frame[candidate]
    direct = []
    ridge = []
    for phase in ("VAL", "HIST", "LATE", "FRESH"):
        direct.append(q.direct_metrics(stock_frame, oriented, f"target_norm_h{horizon}", phase))
        start, end = q.PHASES[phase]
        train = stock_frame[
            (stock_frame[f"target_end_date_h{horizon}"] < start)
            & (stock_frame["date"] >= pd.Timestamp("2008-01-01"))
        ]
        test = stock_frame[stock_frame["date"].between(start, end)]
        pred = fit_predict_pair(
            train, test, stock_parent_cols, oriented, f"target_norm_h{horizon}", "ridge"
        )
        summary = q.summarize_model_predictions(pred, phase)
        ridge.append(summary)
    stock_frame.drop(columns=[oriented], inplace=True)
    direct_map = {x["phase"]: x for x in direct}
    ridge_map = {x["phase"]: x for x in ridge}
    historical_positive = all(
        np.isfinite(direct_map[p]["mean_product_tsic"])
        and direct_map[p]["mean_product_tsic"] > 0
        and np.isfinite(ridge_map[p]["gain_pct"])
        and ridge_map[p]["gain_pct"] > 0
        for p in ("VAL", "HIST")
    )
    return {
        "universe": "30 current liquid US stocks; survivorship-limited domain transfer",
        "direct": direct,
        "ridge_normalized_increment": ridge,
        "val_hist_positive": bool(historical_positive),
    }


def write_report(payload: dict) -> None:
    lines = [
        "# Quantum open-system factor search — strict fast protocol",
        "",
        "Quantum theory is used as a representation and operator-design language. The study does not assert that markets are literal quantum systems.",
        "",
        f"- futures proxy assets: **{payload['data']['futures_assets']}**",
        f"- futures rows: **{payload['data']['futures_rows']}**",
        f"- stock transfer assets: **{payload['data']['stock_assets']}**",
        f"- candidate declarations: **{payload['candidate_family_size']}**",
        f"- selected finalists: **{len(payload['finalists'])}**",
        f"- historical default-off admissions: **{len(payload['admitted_historical_default_off'])}**",
        f"- production independent Alpha admissions: **{len(payload['admitted_production_independent_alpha'])}**",
        "",
        "## Selection contract",
        "",
        "Orientation is fixed on 2008–2013. A candidate must then keep positive direct product-level IC, positive IC against an out-of-sample strong-parent residual, and positive second-half scalar residual gain in both 2014–2015 and 2016–2018. Only one span per mechanism and at most two mechanisms per horizon enter the outer test.",
        "",
    ]
    for item in payload["finalists"]:
        lines.extend([
            f"## `{item['candidate']}` — {item['horizon']} sessions",
            "",
            f"- mechanism: `{item['mechanism']}`",
            f"- development selection: `{item['selection_note']}`",
            f"- stock domain transfer VAL/HIST positive: **{item['stock_domain_transfer']['val_hist_positive']}**",
            f"- historical default-off admitted: **{item['historical_default_off_factor']}**",
            f"- production independent Alpha: **{item['production_independent_alpha']}**",
            f"- family max-T p: `{item['multiplicity'].get('family_maxT_p')}`",
            "",
            "| Period | Futures TSIC | Futures CSIC | Positive products |",
            "|---|---:|---:|---:|",
        ])
        for d in item["direct"]:
            lines.append(
                f"| {d['phase']} | {d['mean_product_tsic']:.6f} | {d['mean_daily_csic']:.6f} | {d['positive_product_share']:.3f} |"
            )
        lines.extend([
            "",
            "| Target | Model | Period | MSE gain % | Block low | Block high |",
            "|---|---|---|---:|---:|---:|",
        ])
        for m in item["model_increment"]:
            lines.append(
                f"| {m['target_kind']} | {m['model_type']} | {m['phase']} | {m['gain_pct']:.6f} | {m['lo']:.6f} | {m['hi']:.6f} |"
            )
        lines.extend([
            "",
            "| Stock period | TSIC | Ridge normalized MSE gain % |",
            "|---|---:|---:|",
        ])
        sd = {x["phase"]: x for x in item["stock_domain_transfer"]["direct"]}
        sm = {x["phase"]: x for x in item["stock_domain_transfer"]["ridge_normalized_increment"]}
        for phase in ("VAL", "HIST", "LATE", "FRESH"):
            lines.append(
                f"| {phase} | {sd[phase]['mean_product_tsic']:.6f} | {sm[phase]['gain_pct']:.6f} |"
            )
        lines.append("")
    lines.extend([
        "## Boundaries",
        "",
        "The futures data are public continuous-contract proxies and can embed vendor roll conventions. The stock transfer panel is a present-day liquid-name universe and therefore survivorship-limited. Costed accounts are fixed-capital diagnostics without contract multipliers, margin, exchange fees, queueing, impact or capacity. Any object admitted here must remain default-off until replayed on the private point-in-time roll-safe production panel.",
        "",
        "The post-2026-09-11 interval is displayed separately and is too short to erase the historical adaptive-search debt. Production independent Alpha remains false unless a genuinely untouched interval, canonical data and execution-aware account all pass.",
    ])
    (OUT / "REPORT_FAST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    futures_panel, futures_coverage = q.download_panel()
    futures_frame, candidates, parent_cols = q.build_features(futures_panel)
    for h in q.HORIZONS:
        invalid = futures_frame[f"target_raw_h{h}"].abs() > 0.35 * math.sqrt(h)
        futures_frame.loc[
            invalid,
            [f"target_raw_h{h}", f"target_norm_h{h}", f"target_open_h{h}"],
        ] = np.nan

    screen = fast_screen(futures_frame, candidates, parent_cols)
    finalists = fast_select(screen)

    stock_panel, stock_coverage = download_stock_panel()
    if not stock_panel.empty:
        stock_frame, _, stock_parent_cols = q.build_features(stock_panel)
        for h in q.HORIZONS:
            invalid = stock_frame[f"target_raw_h{h}"].abs() > 0.35 * math.sqrt(h)
            stock_frame.loc[
                invalid,
                [f"target_raw_h{h}", f"target_norm_h{h}", f"target_open_h{h}"],
            ] = np.nan
    else:
        stock_frame = pd.DataFrame()
        stock_parent_cols = []

    results = []
    for _, row in finalists.iterrows():
        item = q.evaluate_finalist(futures_frame, screen, row, parent_cols)
        if stock_frame.empty or row["candidate"] not in stock_frame.columns:
            transfer = {
                "universe": "unavailable",
                "direct": [],
                "ridge_normalized_increment": [],
                "val_hist_positive": False,
            }
        else:
            transfer = stock_transfer(
                stock_frame,
                stock_parent_cols,
                str(row["candidate"]),
                float(row["orientation"]),
                int(row["horizon"]),
            )
        item["stock_domain_transfer"] = transfer
        # Domain transfer is part of the historical admission gate. Production
        # remains false because public continuous proxies are not a roll-safe,
        # execution-complete production account.
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
    winner = None
    if admitted:
        def score(item: dict) -> float:
            values = [
                x["gain_pct"]
                for x in item["model_increment"]
                if x["phase"] in {"VAL", "HIST"}
                and x["target_kind"] == "norm"
                and x["model_type"] == "ridge"
                and np.isfinite(x["gain_pct"])
            ]
            return min(values) if values else -np.inf
        winner = max(admitted, key=score)
    hourly = q.attempt_hourly_analogue(winner) if winner else {"status": "not_run_no_daily_admission"}

    payload = {
        "schema_version": 2,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "study": "quantum_open_system_measurement_pointer_search_strict_fast",
        "repository_target": "ZZZZYANWANG/Trade",
        "data": {
            "futures_provider": "Yahoo Finance via yfinance",
            "continuous_futures_proxy": True,
            "futures_assets": int(futures_coverage["accepted"].sum()),
            "futures_rows": int(len(futures_frame)),
            "futures_start": str(futures_frame["date"].min().date()),
            "futures_end": str(futures_frame["date"].max().date()),
            "stock_provider": "Yahoo Finance via yfinance",
            "stock_assets": int(stock_coverage["accepted"].sum()) if len(stock_coverage) else 0,
            "stock_rows": int(len(stock_frame)),
            "fresh_cutoff": "2026-09-11",
        },
        "candidate_family_size": int(len(candidates) * len(q.HORIZONS)),
        "parent_feature_count": int(len(parent_cols)),
        "selection_protocol": {
            "orientation": "DEV_A only",
            "replication": ["DEV_B1", "DEV_B2"],
            "outer": ["VAL", "HIST", "LATE", "FRESH"],
            "screen": "direct IC plus strong-parent residual IC plus second-half scalar residual gain",
            "diversity": "one span per horizon-mechanism; max two per horizon",
            "raw_quantum_coordinates_in_parent_set": True,
            "outer_models": ["Ridge", "HistGradientBoosting"],
            "heldout_assets": "three deterministic product folds",
            "stock_domain_transfer": "formula and orientation frozen on futures development",
            "multiplicity": "moving-block family max-T",
        },
        "finalists": results,
        "hourly_homologue": hourly,
        "admitted_historical_default_off": [x["candidate"] for x in admitted],
        "admitted_production_independent_alpha": [],
    }
    (OUT / "RESULTS_FAST.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    write_report(payload)
    print(json.dumps({
        "futures_assets": payload["data"]["futures_assets"],
        "stock_assets": payload["data"]["stock_assets"],
        "candidate_declarations": payload["candidate_family_size"],
        "finalists": len(results),
        "historical_default_off_admissions": payload["admitted_historical_default_off"],
        "production_admissions": [],
        "hourly_status": hourly.get("status"),
    }, indent=2))


if __name__ == "__main__":
    main()
