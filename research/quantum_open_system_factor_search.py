from __future__ import annotations

import json
import math
import os
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "quantum_open_system_results"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20261121
RNG = np.random.default_rng(SEED)
EPS = 1e-12

# Public continuous-futures proxies. The study treats them as a transportability
# panel, not as a production roll implementation.
TICKERS = {
    "ES=F": "equity_index",
    "NQ=F": "equity_index",
    "YM=F": "equity_index",
    "RTY=F": "equity_index",
    "ZB=F": "rates",
    "ZN=F": "rates",
    "ZF=F": "rates",
    "ZT=F": "rates",
    "GC=F": "metal",
    "SI=F": "metal",
    "HG=F": "metal",
    "PL=F": "metal",
    "PA=F": "metal",
    "CL=F": "energy",
    "BZ=F": "energy",
    "NG=F": "energy",
    "HO=F": "energy",
    "RB=F": "energy",
    "ZC=F": "agriculture",
    "ZW=F": "agriculture",
    "ZS=F": "agriculture",
    "ZM=F": "agriculture",
    "ZL=F": "agriculture",
    "KC=F": "soft",
    "SB=F": "soft",
    "CC=F": "soft",
    "CT=F": "soft",
    "LE=F": "livestock",
    "HE=F": "livestock",
    "6E=F": "fx",
    "6J=F": "fx",
    "6B=F": "fx",
    "6A=F": "fx",
    "6C=F": "fx",
    "DX-Y.NYB": "fx",
}

START = "2007-01-01"
END = None
HORIZONS = (1, 5, 10, 20)
SPANS = (10, 20, 40, 80)

PHASES = {
    "DEV_A": (pd.Timestamp("2008-01-01"), pd.Timestamp("2013-12-31")),
    "DEV_B1": (pd.Timestamp("2014-01-01"), pd.Timestamp("2015-12-31")),
    "DEV_B2": (pd.Timestamp("2016-01-01"), pd.Timestamp("2018-12-31")),
    "VAL": (pd.Timestamp("2019-01-01"), pd.Timestamp("2021-12-31")),
    "HIST": (pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31")),
    "LATE": (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-09-11")),
    "FRESH": (pd.Timestamp("2026-09-12"), pd.Timestamp("2099-12-31")),
}


@dataclass(frozen=True)
class CandidateMeta:
    candidate: str
    mechanism: str
    span: int
    formula_role: str


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        # Single-ticker yfinance downloads normally have price fields on level 0.
        if set([str(x).lower() for x in out.columns.get_level_values(0)]) & {
            "open", "high", "low", "close", "adj close", "volume"
        }:
            out.columns = out.columns.get_level_values(0)
        else:
            out.columns = out.columns.get_level_values(-1)
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def download_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    coverage: list[dict] = []
    for ticker, sector in TICKERS.items():
        error = None
        data = pd.DataFrame()
        for attempt in range(3):
            try:
                data = yf.download(
                    ticker,
                    start=START,
                    end=END,
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                    threads=False,
                    timeout=30,
                )
                data = _flatten_columns(data)
                if len(data) >= 750:
                    break
            except Exception as exc:  # pragma: no cover - external data path
                error = repr(exc)
            time.sleep(1.5 * (attempt + 1))
        needed = {"open", "high", "low", "close", "volume"}
        if data.empty or not needed.issubset(data.columns):
            coverage.append({
                "ticker": ticker,
                "sector": sector,
                "rows": int(len(data)),
                "start": None,
                "end": None,
                "accepted": False,
                "reason": error or "missing_required_columns",
            })
            continue
        data = data.reset_index()
        date_col = "date" if "date" in data.columns else data.columns[0]
        data = data.rename(columns={date_col: "date"})
        data["date"] = pd.to_datetime(data["date"], utc=True).dt.tz_convert(None).dt.normalize()
        for col in needed | {"adj_close"}:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna(subset=["date", "open", "high", "low", "close"])
        data = data[(data[["open", "high", "low", "close"]] > 0).all(axis=1)]
        data = data.sort_values("date").drop_duplicates("date", keep="last")
        accepted = len(data) >= 750 and data["date"].nunique() >= 750
        coverage.append({
            "ticker": ticker,
            "sector": sector,
            "rows": int(len(data)),
            "start": str(data["date"].min().date()) if len(data) else None,
            "end": str(data["date"].max().date()) if len(data) else None,
            "accepted": bool(accepted),
            "reason": None if accepted else "insufficient_history",
        })
        if not accepted:
            continue
        data["ticker"] = ticker
        data["sector"] = sector
        frames.append(data[["date", "ticker", "sector", "open", "high", "low", "close", "volume"]])
    coverage_df = pd.DataFrame(coverage)
    coverage_df.to_csv(OUT / "data_coverage.csv", index=False)
    if not frames:
        raise RuntimeError("No market series passed the minimum history gate")
    panel = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    return panel, coverage_df


def strict_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    minp = min_periods or max(20, window // 3)
    prior = series.shift(1)
    mean = prior.rolling(window, min_periods=minp).mean()
    std = prior.rolling(window, min_periods=minp).std(ddof=0)
    return (series - mean) / (std + EPS)


def rolling_rms(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    minp = min_periods or max(10, window // 2)
    return np.sqrt(series.pow(2).rolling(window, min_periods=minp).mean())


def rank_pct(series: pd.Series) -> pd.Series:
    return series.rank(pct=True, method="average") * 2.0 - 1.0


def build_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[CandidateMeta], list[str]]:
    outputs: list[pd.DataFrame] = []
    candidate_meta: dict[str, CandidateMeta] = {}
    base_parent_cols = [
        "ret1_z", "mom5", "mom10", "mom20", "mom60",
        "gap_z", "body_z", "close_location", "body_fraction",
        "range_z", "volume_z", "range_level", "volume_change",
        "path_efficiency20", "signed_efficiency20", "autocorr20",
        "down_up_vol_ratio20", "ret1_range_interaction", "ret1_volume_interaction",
        "trend_range_interaction", "trend_volume_interaction",
    ]

    for ticker, g0 in panel.groupby("ticker", sort=False):
        g = g0.sort_values("date").copy()
        logc = np.log(g["close"].astype(float))
        logo = np.log(g["open"].astype(float))
        logh = np.log(g["high"].astype(float))
        logl = np.log(g["low"].astype(float))
        ret1 = logc.diff()
        gap = logo - logc.shift(1)
        body = logc - logo
        log_range = (logh - logl).clip(lower=0)
        rv20 = rolling_rms(ret1, 20, 12)
        range_level = log_range / (rv20 + EPS)
        denom = (g["high"] - g["low"]).replace(0, np.nan)
        close_location = ((2.0 * g["close"] - g["high"] - g["low"]) / denom).clip(-1, 1)
        body_fraction = ((g["close"] - g["open"]) / denom).clip(-1, 1)
        log_volume = np.log1p(g["volume"].where(g["volume"] > 0))
        volume_change = log_volume.diff()
        range_z = strict_zscore(log_range, 63, 30).clip(-6, 6)
        volume_z = strict_zscore(log_volume, 63, 30).clip(-6, 6).fillna(0.0)
        gap_z = (gap / (rv20 + EPS)).clip(-6, 6)
        body_z = (body / (rv20 + EPS)).clip(-6, 6)
        ret1_z = (ret1 / (rv20 + EPS)).clip(-6, 6)

        g["ret1"] = ret1
        g["rv20"] = rv20
        g["ret1_z"] = ret1_z
        g["gap_z"] = gap_z
        g["body_z"] = body_z
        g["close_location"] = close_location
        g["body_fraction"] = body_fraction
        g["range_z"] = range_z
        g["volume_z"] = volume_z
        g["range_level"] = range_level.clip(0, 10)
        g["volume_change"] = volume_change.clip(-6, 6)

        abs_sum20 = ret1.abs().rolling(20, min_periods=12).sum()
        sum20 = ret1.rolling(20, min_periods=12).sum()
        g["path_efficiency20"] = (sum20.abs() / (abs_sum20 + EPS)).clip(0, 1)
        g["signed_efficiency20"] = (sum20 / (abs_sum20 + EPS)).clip(-1, 1)
        g["autocorr20"] = ret1.rolling(20, min_periods=15).corr(ret1.shift(1)).clip(-1, 1)
        upv = ret1.clip(lower=0).pow(2).rolling(20, min_periods=12).mean()
        dnv = (-ret1.clip(upper=0)).pow(2).rolling(20, min_periods=12).mean()
        g["down_up_vol_ratio20"] = np.log((dnv + EPS) / (upv + EPS)).clip(-6, 6)

        for h in (5, 10, 20, 60):
            mom = logc - logc.shift(h)
            scale = rv20 * math.sqrt(h)
            g[f"mom{h}"] = (mom / (scale + EPS)).clip(-8, 8)
        g["ret1_range_interaction"] = ret1_z * range_z.clip(lower=0)
        g["ret1_volume_interaction"] = ret1_z * volume_z.clip(lower=0)
        g["trend_range_interaction"] = g["mom20"] * range_z.clip(lower=0)
        g["trend_volume_interaction"] = g["mom20"] * volume_z.clip(lower=0)

        # Completed-bar qubit state on the Bloch ball.
        bx = np.tanh(body_z / 2.0)
        by = np.tanh(gap_z / 2.0)
        bz = close_location.fillna(0.0).clip(-0.999, 0.999)
        bnorm = np.sqrt(bx.pow(2) + by.pow(2) + bz.pow(2))
        scale = np.maximum(1.0, bnorm / 0.999)
        bx = bx / scale
        by = by / scale
        bz = bz / scale
        obs_norm2 = (bx.pow(2) + by.pow(2) + bz.pow(2)).clip(0, 0.999999)
        mstrength = np.tanh(
            0.45 * range_z.clip(lower=0).fillna(0.0)
            + 0.25 * volume_z.clip(lower=0).fillna(0.0)
        ).clip(0, 1)
        current = np.tanh(ret1_z / 2.0)
        trend = np.tanh(g["mom20"] / 2.0)

        for span in SPANS:
            minp = max(8, span // 2)
            px = bx.ewm(span=span, min_periods=minp, adjust=False).mean().shift(1)
            py = by.ewm(span=span, min_periods=minp, adjust=False).mean().shift(1)
            pz = bz.ewm(span=span, min_periods=minp, adjust=False).mean().shift(1)
            prior_norm2 = (px.pow(2) + py.pow(2) + pz.pow(2)).clip(0, 0.999999)
            dot = bx * px + by * py + bz * pz
            fidelity = 0.5 * (
                1.0
                + dot
                + np.sqrt((1.0 - obs_norm2).clip(lower=0))
                * np.sqrt((1.0 - prior_norm2).clip(lower=0))
            )
            fidelity = fidelity.clip(0, 1)
            bures2 = (2.0 * (1.0 - np.sqrt(fidelity))).clip(0, 2)
            ix = bx - px
            iy = by - py
            iz = bz - pz
            innovation_norm = np.sqrt(ix.pow(2) + iy.pow(2) + iz.pow(2))
            cx = py * bz - pz * by
            cy = pz * bx - px * bz
            cz = px * by - py * bx
            comm_norm = np.sqrt(cx.pow(2) + cy.pow(2) + cz.pow(2))
            purity_gain = 0.5 * (obs_norm2 - prior_norm2)
            axis_projection = ix * px + iy * py + iz * pz
            transverse_projection = ix * (-py) + iy * px
            dissipation_gap = comm_norm - innovation_norm

            state_cols = {
                f"q_fidelity_s{span}": fidelity,
                f"q_bures2_s{span}": bures2,
                f"q_purity_gain_s{span}": purity_gain,
                f"q_commutator_z_s{span}": cz,
                f"q_commutator_norm_s{span}": comm_norm,
                f"q_innovation_norm_s{span}": innovation_norm,
                f"q_axis_projection_s{span}": axis_projection,
            }
            for name, value in state_cols.items():
                g[name] = value

            definitions = {
                f"measurement_backaction_reversal_s{span}": (
                    -current * bures2 * (0.20 + mstrength) * (1.0 + purity_gain.clip(lower=0))
                ),
                f"pointer_leakage_trend_reversal_s{span}": (
                    -trend * bures2 * (0.20 + mstrength) * (1.0 + purity_gain.clip(lower=0))
                ),
                f"coherent_rotation_current_s{span}": (
                    cz * (0.20 + mstrength) * np.sqrt(fidelity.clip(lower=0))
                ),
                f"coherent_rotation_return_s{span}": (
                    current * cz * (0.20 + mstrength)
                ),
                f"zeno_pointer_persistence_s{span}": (
                    trend * fidelity * (1.0 - 0.75 * mstrength) * (1.0 - purity_gain.clip(lower=0))
                ),
                f"purity_backaction_trend_s{span}": (
                    -trend * purity_gain * bures2 * (0.20 + mstrength)
                ),
                f"filter_innovation_axis_s{span}": (
                    axis_projection * (0.20 + mstrength)
                ),
                f"filter_innovation_transverse_s{span}": (
                    transverse_projection * (0.20 + mstrength)
                ),
                f"hamiltonian_dissipation_gap_s{span}": (
                    trend * dissipation_gap * (0.20 + mstrength)
                ),
            }
            mechanism_map = {
                "measurement_backaction_reversal": "measurement_backaction",
                "pointer_leakage_trend_reversal": "pointer_state_leakage",
                "coherent_rotation_current": "commutator_current",
                "coherent_rotation_return": "commutator_current",
                "zeno_pointer_persistence": "zeno_persistence",
                "purity_backaction_trend": "purity_backaction",
                "filter_innovation_axis": "quantum_filter_innovation",
                "filter_innovation_transverse": "quantum_filter_innovation",
                "hamiltonian_dissipation_gap": "open_system_generator_split",
            }
            for name, value in definitions.items():
                g[name] = pd.Series(value, index=g.index).replace([np.inf, -np.inf], np.nan).clip(-20, 20)
                root = name.rsplit("_s", 1)[0]
                candidate_meta[name] = CandidateMeta(
                    candidate=name,
                    mechanism=mechanism_map[root],
                    span=span,
                    formula_role=root,
                )

        # Raw quantum-state coordinates are deliberately included as controls.
        outputs.append(g)

    frame = pd.concat(outputs, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)

    quantum_state_controls = [
        c for c in frame.columns
        if c.startswith("q_") and not c.startswith("quantum_")
    ]
    parent_cols = base_parent_cols + quantum_state_controls

    for h in HORIZONS:
        target_col = f"target_raw_h{h}"
        target_norm_col = f"target_norm_h{h}"
        open_col = f"target_open_h{h}"
        end_col = f"target_end_date_h{h}"
        frame[target_col] = np.nan
        frame[target_norm_col] = np.nan
        frame[open_col] = np.nan
        frame[end_col] = pd.NaT
        for ticker, idx in frame.groupby("ticker", sort=False).groups.items():
            loc = np.asarray(list(idx), dtype=int)
            gg = frame.loc[loc].sort_values("date")
            raw = np.log(gg["close"].shift(-h)) - np.log(gg["close"])
            norm = raw / (gg["rv20"] * math.sqrt(h) + EPS)
            open_ret = np.log(gg["open"].shift(-(h + 1))) - np.log(gg["open"].shift(-1))
            end_date = gg["date"].shift(-h)
            frame.loc[gg.index, target_col] = raw.values
            frame.loc[gg.index, target_norm_col] = norm.values
            frame.loc[gg.index, open_col] = open_ret.values
            frame.loc[gg.index, end_col] = end_date.values

    return frame, list(candidate_meta.values()), parent_cols


def phase_mask(frame: pd.DataFrame, phase: str) -> pd.Series:
    start, end = PHASES[phase]
    return frame["date"].between(start, end)


def finite_pair(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    xa = pd.to_numeric(x, errors="coerce").to_numpy(float)
    ya = pd.to_numeric(y, errors="coerce").to_numpy(float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    return xa[mask], ya[mask]


def safe_spearman(x: pd.Series, y: pd.Series, min_n: int = 30) -> float:
    xa, ya = finite_pair(x, y)
    if len(xa) < min_n or np.nanstd(xa) < EPS or np.nanstd(ya) < EPS:
        return np.nan
    return float(spearmanr(xa, ya).correlation)


def direct_metrics(frame: pd.DataFrame, candidate: str, target: str, phase: str) -> dict:
    sub = frame.loc[phase_mask(frame, phase), ["date", "ticker", candidate, target]].copy()
    product_ics = []
    for _, g in sub.groupby("ticker"):
        value = safe_spearman(g[candidate], g[target], min_n=40)
        if np.isfinite(value):
            product_ics.append(value)
    daily_ics = []
    for _, g in sub.groupby("date"):
        value = safe_spearman(g[candidate], g[target], min_n=8)
        if np.isfinite(value):
            daily_ics.append(value)
    return {
        "phase": phase,
        "n_rows": int(sub[[candidate, target]].dropna().shape[0]),
        "products": int(len(product_ics)),
        "dates": int(len(daily_ics)),
        "mean_product_tsic": float(np.mean(product_ics)) if product_ics else np.nan,
        "median_product_tsic": float(np.median(product_ics)) if product_ics else np.nan,
        "positive_product_share": float(np.mean(np.asarray(product_ics) > 0)) if product_ics else np.nan,
        "mean_daily_csic": float(np.mean(daily_ics)) if daily_ics else np.nan,
        "positive_date_share": float(np.mean(np.asarray(daily_ics) > 0)) if daily_ics else np.nan,
    }


def _model_pipeline(model_type: str, numeric_cols: list[str], categorical_cols: list[str]):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    pre = ColumnTransformer([
        ("num", numeric, numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), categorical_cols),
    ])
    if model_type == "ridge":
        model = Ridge(alpha=25.0, fit_intercept=True)
    elif model_type == "hgb":
        # HistGradientBoosting does not accept sparse one-hot output. Use sector only
        # through deterministic integer features added by the caller and numeric path.
        pre = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]), numeric_cols),
        ])
        model = HistGradientBoostingRegressor(
            learning_rate=0.045,
            max_iter=110,
            max_leaf_nodes=15,
            min_samples_leaf=80,
            l2_regularization=0.5,
            random_state=SEED,
        )
    else:
        raise ValueError(model_type)
    return Pipeline([("pre", pre), ("model", model)])


def fit_predict_pair(
    train: pd.DataFrame,
    test: pd.DataFrame,
    parent_cols: list[str],
    candidate: str,
    target: str,
    model_type: str,
) -> pd.DataFrame:
    categorical = ["ticker", "sector"] if model_type == "ridge" else []
    train_cols = parent_cols + categorical + [target]
    test_cols = parent_cols + categorical + [target]
    tr = train[train_cols + [candidate]].copy()
    te = test[test_cols + [candidate, "date", "ticker"]].copy()
    tr = tr[np.isfinite(pd.to_numeric(tr[target], errors="coerce"))]
    te = te[np.isfinite(pd.to_numeric(te[target], errors="coerce"))]
    if len(tr) < 3000 or len(te) < 100:
        return pd.DataFrame()
    parent_pipe = _model_pipeline(model_type, parent_cols, categorical)
    joint_pipe = _model_pipeline(model_type, parent_cols + [candidate], categorical)
    parent_pipe.fit(tr[parent_cols + categorical], tr[target].astype(float))
    joint_pipe.fit(tr[parent_cols + [candidate] + categorical], tr[target].astype(float))
    out = te[["date", "ticker", target]].copy()
    out["parent_prediction"] = parent_pipe.predict(te[parent_cols + categorical])
    out["joint_prediction"] = joint_pipe.predict(te[parent_cols + [candidate] + categorical])
    out["parent_loss"] = (out[target] - out["parent_prediction"]) ** 2
    out["joint_loss"] = (out[target] - out["joint_prediction"]) ** 2
    return out


def fixed_dev_fold_gain(
    frame: pd.DataFrame,
    parent_cols: list[str],
    candidate: str,
    horizon: int,
    test_start: str,
    test_end: str,
) -> float:
    target = f"target_norm_h{horizon}"
    start = pd.Timestamp(test_start)
    end = pd.Timestamp(test_end)
    train = frame[(frame[f"target_end_date_h{horizon}"] < start) & (frame["date"] >= pd.Timestamp("2008-01-01"))]
    test = frame[frame["date"].between(start, end)]
    pred = fit_predict_pair(train, test, parent_cols, candidate, target, "ridge")
    if pred.empty:
        return np.nan
    base = float(pred["parent_loss"].mean())
    joint = float(pred["joint_loss"].mean())
    return 100.0 * (base - joint) / (base + EPS)


def screen_candidates(
    frame: pd.DataFrame,
    candidates: list[CandidateMeta],
    parent_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        target = f"target_norm_h{h}"
        for meta in candidates:
            a = direct_metrics(frame, meta.candidate, target, "DEV_A")
            b1 = direct_metrics(frame, meta.candidate, target, "DEV_B1")
            b2 = direct_metrics(frame, meta.candidate, target, "DEV_B2")
            base_ic = a["mean_product_tsic"]
            orientation = 1.0 if np.isfinite(base_ic) and base_ic >= 0 else -1.0
            oriented = meta.candidate
            temp_col = f"__oriented_{meta.candidate}"
            frame[temp_col] = orientation * frame[meta.candidate]
            gain1 = fixed_dev_fold_gain(frame, parent_cols, temp_col, h, "2014-01-01", "2015-12-31")
            gain2 = fixed_dev_fold_gain(frame, parent_cols, temp_col, h, "2016-01-01", "2018-12-31")
            frame.drop(columns=[temp_col], inplace=True)
            rows.append({
                **asdict(meta),
                "horizon": h,
                "orientation": orientation,
                "dev_a_tsic": orientation * a["mean_product_tsic"],
                "dev_b1_tsic": orientation * b1["mean_product_tsic"],
                "dev_b2_tsic": orientation * b2["mean_product_tsic"],
                "dev_a_csic": orientation * a["mean_daily_csic"],
                "dev_b1_csic": orientation * b1["mean_daily_csic"],
                "dev_b2_csic": orientation * b2["mean_daily_csic"],
                "dev_fold1_ridge_gain_pct": gain1,
                "dev_fold2_ridge_gain_pct": gain2,
            })
    out = pd.DataFrame(rows)
    out["min_dev_tsic"] = out[["dev_a_tsic", "dev_b1_tsic", "dev_b2_tsic"]].min(axis=1)
    out["min_dev_csic"] = out[["dev_a_csic", "dev_b1_csic", "dev_b2_csic"]].min(axis=1)
    out["min_dev_gain"] = out[["dev_fold1_ridge_gain_pct", "dev_fold2_ridge_gain_pct"]].min(axis=1)
    out["dev_pass"] = (
        (out["min_dev_tsic"] > 0.0025)
        & (out["min_dev_gain"] > 0.0)
        & (out[["dev_a_tsic", "dev_b1_tsic", "dev_b2_tsic"]].notna().all(axis=1))
    )
    out.to_csv(OUT / "candidate_screen.csv", index=False)
    return out


def select_finalists(screen: pd.DataFrame, max_finalists: int = 8) -> pd.DataFrame:
    passed = screen[screen["dev_pass"]].copy()
    if passed.empty:
        passed = screen.sort_values(["min_dev_gain", "min_dev_tsic"], ascending=False).head(max_finalists).copy()
        passed["selection_note"] = "diagnostic_only_no_full_development_pass"
    else:
        passed["selection_note"] = "passed_development_gate"
    # One representative per horizon and mechanism prevents near-duplicate spans
    # from consuming separate discovery budgets.
    passed = passed.sort_values(
        ["horizon", "mechanism", "min_dev_gain", "min_dev_tsic"],
        ascending=[True, True, False, False],
    ).groupby(["horizon", "mechanism"], as_index=False).head(1)
    passed = passed.sort_values(["min_dev_gain", "min_dev_tsic"], ascending=False).head(max_finalists)
    passed.to_csv(OUT / "selected_finalists.csv", index=False)
    return passed


def annual_oof_predictions(
    frame: pd.DataFrame,
    parent_cols: list[str],
    candidate: str,
    horizon: int,
    target_kind: str,
    model_type: str,
) -> pd.DataFrame:
    target = f"target_{target_kind}_h{horizon}"
    pieces = []
    max_year = int(frame["date"].dt.year.max())
    for year in range(2019, max_year + 1):
        start = pd.Timestamp(year=year, month=1, day=1)
        end = pd.Timestamp(year=year, month=12, day=31)
        train = frame[(frame[f"target_end_date_h{horizon}"] < start) & (frame["date"] >= pd.Timestamp("2008-01-01"))]
        test = frame[frame["date"].between(start, end)]
        pred = fit_predict_pair(train, test, parent_cols, candidate, target, model_type)
        if not pred.empty:
            pred["fit_cutoff"] = str(start.date())
            pred["year"] = year
            pieces.append(pred)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def moving_block_interval(values: Iterable[float], block: int = 20, draws: int = 1200) -> dict:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < max(20, block):
        return {"mean": float(np.mean(x)) if len(x) else np.nan, "lo": np.nan, "hi": np.nan, "p_nonpositive": np.nan, "n": int(len(x))}
    n = len(x)
    starts = np.arange(n)
    samples = np.empty(draws)
    for b in range(draws):
        chosen = []
        while len(chosen) < n:
            s = int(RNG.choice(starts))
            chosen.extend([x[(s + j) % n] for j in range(block)])
        samples[b] = np.mean(chosen[:n])
    return {
        "mean": float(np.mean(x)),
        "lo": float(np.quantile(samples, 0.025)),
        "hi": float(np.quantile(samples, 0.975)),
        "p_nonpositive": float(np.mean(samples <= 0)),
        "n": int(n),
    }


def summarize_model_predictions(pred: pd.DataFrame, phase: str) -> dict:
    if pred.empty:
        return {"phase": phase, "n": 0, "gain_pct": np.nan, "lo": np.nan, "hi": np.nan, "p_nonpositive": np.nan}
    start, end = PHASES[phase]
    sub = pred[pred["date"].between(start, end)].copy()
    if sub.empty:
        return {"phase": phase, "n": 0, "gain_pct": np.nan, "lo": np.nan, "hi": np.nan, "p_nonpositive": np.nan}
    base = float(sub["parent_loss"].mean())
    joint = float(sub["joint_loss"].mean())
    gain = 100.0 * (base - joint) / (base + EPS)
    daily = sub.groupby("date").agg(parent_loss=("parent_loss", "mean"), joint_loss=("joint_loss", "mean"))
    daily["gain_num"] = daily["parent_loss"] - daily["joint_loss"]
    ci = moving_block_interval(daily["gain_num"], block=20)
    scale = 100.0 / (base + EPS)
    return {
        "phase": phase,
        "n": int(len(sub)),
        "dates": int(daily.shape[0]),
        "gain_pct": gain,
        "lo": ci["lo"] * scale if np.isfinite(ci["lo"]) else np.nan,
        "hi": ci["hi"] * scale if np.isfinite(ci["hi"]) else np.nan,
        "p_nonpositive": ci["p_nonpositive"],
    }


def reconstruction_scores(
    frame: pd.DataFrame,
    parent_cols: list[str],
    candidate: str,
    phase: str,
) -> dict:
    start, end = PHASES[phase]
    test = frame[frame["date"].between(start, end)].copy()
    train = frame[frame["date"] < start].copy()
    cols = parent_cols + ["ticker", "sector"]
    tr = train[cols + [candidate]].dropna(subset=[candidate])
    te = test[cols + [candidate]].dropna(subset=[candidate])
    if len(tr) < 3000 or len(te) < 100:
        return {"phase": phase, "ridge_r2": np.nan, "hgb_r2": np.nan, "n": int(len(te))}
    ridge = _model_pipeline("ridge", parent_cols, ["ticker", "sector"])
    ridge.fit(tr[cols], tr[candidate])
    pr = ridge.predict(te[cols])
    hgb = _model_pipeline("hgb", parent_cols, [])
    hgb.fit(tr[parent_cols], tr[candidate])
    ph = hgb.predict(te[parent_cols])
    return {
        "phase": phase,
        "ridge_r2": float(r2_score(te[candidate], pr)),
        "hgb_r2": float(r2_score(te[candidate], ph)),
        "n": int(len(te)),
    }


def heldout_asset_gain(
    frame: pd.DataFrame,
    parent_cols: list[str],
    candidate: str,
    horizon: int,
    phase: str,
) -> dict:
    target = f"target_norm_h{horizon}"
    start, end = PHASES[phase]
    phase_rows = frame[frame["date"].between(start, end)].copy()
    products = sorted(phase_rows["ticker"].dropna().unique())
    if len(products) < 12:
        return {"phase": phase, "gain_pct": np.nan, "n": 0, "folds": 0}
    fold_map = {p: i % 3 for i, p in enumerate(products)}
    pieces = []
    for fold in range(3):
        held = [p for p in products if fold_map[p] == fold]
        train = frame[(frame[f"target_end_date_h{horizon}"] < start) & (~frame["ticker"].isin(held))]
        test = phase_rows[phase_rows["ticker"].isin(held)]
        # No asset identity features are used in transfer evaluation.
        tr = train[parent_cols + [candidate, target]].copy()
        te = test[parent_cols + [candidate, target, "date", "ticker"]].copy()
        tr = tr[np.isfinite(pd.to_numeric(tr[target], errors="coerce"))]
        te = te[np.isfinite(pd.to_numeric(te[target], errors="coerce"))]
        if len(tr) < 3000 or len(te) < 100:
            continue
        parent_pipe = _model_pipeline("ridge", parent_cols, [])
        joint_pipe = _model_pipeline("ridge", parent_cols + [candidate], [])
        parent_pipe.fit(tr[parent_cols], tr[target])
        joint_pipe.fit(tr[parent_cols + [candidate]], tr[target])
        y = te[target].to_numpy(float)
        pieces.append(pd.DataFrame({
            "date": te["date"].values,
            "parent_loss": (y - parent_pipe.predict(te[parent_cols])) ** 2,
            "joint_loss": (y - joint_pipe.predict(te[parent_cols + [candidate]])) ** 2,
        }))
    if not pieces:
        return {"phase": phase, "gain_pct": np.nan, "n": 0, "folds": 0}
    out = pd.concat(pieces, ignore_index=True)
    base = float(out["parent_loss"].mean())
    joint = float(out["joint_loss"].mean())
    return {
        "phase": phase,
        "gain_pct": 100.0 * (base - joint) / (base + EPS),
        "n": int(len(out)),
        "folds": int(len(pieces)),
    }


def costed_nonoverlap_account(
    frame: pd.DataFrame,
    candidate: str,
    horizon: int,
    phase: str,
    one_way_cost_bp: float,
) -> dict:
    start, end = PHASES[phase]
    cols = ["date", "ticker", candidate, "rv20", f"target_open_h{horizon}"]
    sub = frame.loc[frame["date"].between(start, end), cols].copy()
    if sub.empty:
        return {"phase": phase, "cost_bp": one_way_cost_bp, "periods": 0}
    sub["signal_z"] = sub.groupby("ticker")[candidate].transform(
        lambda s: (s - s.shift(1).rolling(252, min_periods=80).mean())
        / (s.shift(1).rolling(252, min_periods=80).std(ddof=0) + EPS)
    )
    sub["raw_weight"] = np.tanh(sub["signal_z"] / 2.0) / (sub["rv20"].clip(lower=1e-4))
    records = []
    all_dates = sorted(sub["date"].dropna().unique())
    date_index = {d: i for i, d in enumerate(all_dates)}
    sub["date_idx"] = sub["date"].map(date_index)
    for offset in range(horizon):
        q = sub[(sub["date_idx"] % horizon) == offset].copy()
        previous: dict[str, float] = {}
        for date, g in q.groupby("date", sort=True):
            g = g.dropna(subset=["raw_weight", f"target_open_h{horizon}"])
            if len(g) < 8:
                continue
            weights = g.set_index("ticker")["raw_weight"].astype(float)
            gross = float(weights.abs().sum())
            if gross <= EPS:
                continue
            weights = weights / gross
            returns = g.set_index("ticker")[f"target_open_h{horizon}"].astype(float)
            gross_ret = float((weights * returns.reindex(weights.index)).sum())
            names = set(previous) | set(weights.index)
            turnover = sum(abs(float(weights.get(n, 0.0)) - float(previous.get(n, 0.0))) for n in names)
            net = gross_ret - turnover * one_way_cost_bp * 1e-4
            records.append({"date": date, "offset": offset, "net_return": net, "gross_return": gross_ret, "turnover": turnover})
            previous = weights.to_dict()
    account = pd.DataFrame(records)
    if account.empty:
        return {"phase": phase, "cost_bp": one_way_cost_bp, "periods": 0}
    # Average simultaneous offset sleeves on each exit date.
    daily = account.groupby("date").agg(
        net_return=("net_return", "mean"),
        gross_return=("gross_return", "mean"),
        turnover=("turnover", "mean"),
    )
    ann_factor = 252.0 / max(1, horizon)
    mean = float(daily["net_return"].mean())
    std = float(daily["net_return"].std(ddof=1))
    wealth = (1.0 + daily["net_return"].clip(lower=-0.99)).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    ci = moving_block_interval(daily["net_return"], block=max(5, 40 // max(1, horizon)))
    return {
        "phase": phase,
        "cost_bp": one_way_cost_bp,
        "periods": int(len(daily)),
        "annual_return_linear": mean * ann_factor,
        "sharpe": mean / (std + EPS) * math.sqrt(ann_factor),
        "max_drawdown": float(drawdown.min()),
        "mean_turnover": float(daily["turnover"].mean()),
        "block_mean_lo": ci["lo"],
        "block_mean_hi": ci["hi"],
        "block_p_nonpositive": ci["p_nonpositive"],
    }


def daily_csic_series(frame: pd.DataFrame, candidate: str, target: str, phase: str) -> pd.Series:
    sub = frame.loc[phase_mask(frame, phase), ["date", candidate, target]].copy()
    values = {}
    for date, g in sub.groupby("date"):
        ic = safe_spearman(g[candidate], g[target], min_n=8)
        if np.isfinite(ic):
            values[pd.Timestamp(date)] = ic
    return pd.Series(values, dtype=float).sort_index()


def family_max_t_p(
    frame: pd.DataFrame,
    screen: pd.DataFrame,
    selected_row: pd.Series,
    phase_names: tuple[str, ...] = ("VAL", "HIST"),
    draws: int = 1000,
    block: int = 20,
) -> dict:
    h = int(selected_row["horizon"])
    target = f"target_norm_h{h}"
    same_h = screen[screen["horizon"] == h]
    series = {}
    for _, row in same_h.iterrows():
        name = row["candidate"]
        orientation = float(row["orientation"])
        temp = f"__oriented_{name}"
        frame[temp] = orientation * frame[name]
        parts = [daily_csic_series(frame, temp, target, p) for p in phase_names]
        frame.drop(columns=[temp], inplace=True)
        series[name] = pd.concat(parts).sort_index()
    matrix = pd.DataFrame(series).dropna(how="all")
    if selected_row["candidate"] not in matrix or matrix[selected_row["candidate"]].count() < 30:
        return {"observed": np.nan, "family_maxT_p": np.nan, "dates": int(len(matrix)), "family": int(len(series))}
    matrix = matrix.fillna(0.0)
    observed_means = matrix.mean(axis=0)
    observed = float(observed_means[selected_row["candidate"]])
    centered = matrix - observed_means
    n = len(centered)
    maxima = np.empty(draws)
    arr = centered.to_numpy(float)
    for b in range(draws):
        idx = []
        while len(idx) < n:
            s = int(RNG.integers(0, n))
            idx.extend([(s + j) % n for j in range(block)])
        sample = arr[np.asarray(idx[:n])]
        maxima[b] = np.nanmax(sample.mean(axis=0))
    return {
        "observed": observed,
        "family_maxT_p": float((1 + np.sum(maxima >= observed)) / (draws + 1)),
        "dates": int(n),
        "family": int(len(series)),
    }


def evaluate_finalist(
    frame: pd.DataFrame,
    screen: pd.DataFrame,
    row: pd.Series,
    parent_cols: list[str],
) -> dict:
    candidate = str(row["candidate"])
    h = int(row["horizon"])
    orientation = float(row["orientation"])
    oriented = f"oriented__{candidate}"
    frame[oriented] = orientation * frame[candidate]
    direct = []
    for phase in PHASES:
        direct.append(direct_metrics(frame, oriented, f"target_norm_h{h}", phase))

    model_rows = []
    prediction_paths = []
    for target_kind in ("raw", "norm"):
        for model_type in ("ridge", "hgb"):
            pred = annual_oof_predictions(frame, parent_cols, oriented, h, target_kind, model_type)
            if not pred.empty:
                pred["candidate"] = candidate
                pred["horizon"] = h
                pred["target_kind"] = target_kind
                pred["model_type"] = model_type
                path = OUT / f"predictions__{candidate}__h{h}__{target_kind}__{model_type}.csv.gz"
                pred.to_csv(path, index=False, compression="gzip")
                prediction_paths.append(str(path.name))
            for phase in ("VAL", "HIST", "LATE", "FRESH"):
                summary = summarize_model_predictions(pred, phase)
                summary.update({"target_kind": target_kind, "model_type": model_type})
                model_rows.append(summary)

    reconstruction = [
        reconstruction_scores(frame, parent_cols, oriented, phase)
        for phase in ("VAL", "HIST", "LATE", "FRESH")
    ]
    heldout = [
        heldout_asset_gain(frame, parent_cols, oriented, h, phase)
        for phase in ("VAL", "HIST", "LATE", "FRESH")
    ]
    accounts = [
        costed_nonoverlap_account(frame, oriented, h, phase, cost)
        for phase in ("VAL", "HIST", "LATE", "FRESH")
        for cost in (0.0, 1.0, 2.0, 5.0)
    ]
    multiplicity = family_max_t_p(frame, screen, row)

    model_df = pd.DataFrame(model_rows)
    required = model_df[
        model_df["phase"].isin(["VAL", "HIST"])
        & model_df["target_kind"].isin(["raw", "norm"])
        & model_df["model_type"].isin(["ridge", "hgb"])
    ]
    point_positive = bool(len(required) == 8 and (required["gain_pct"] > 0).all())
    ridge_norm = required[(required["target_kind"] == "norm") & (required["model_type"] == "ridge")]
    interval_positive = bool(len(ridge_norm) == 2 and (ridge_norm["lo"] > 0).all())
    direct_map = {d["phase"]: d for d in direct}
    direction_positive = all(
        np.isfinite(direct_map[p]["mean_product_tsic"])
        and direct_map[p]["mean_product_tsic"] > 0
        for p in ("VAL", "HIST")
    )
    held_map = {d["phase"]: d for d in heldout}
    heldout_positive = all(
        np.isfinite(held_map[p]["gain_pct"]) and held_map[p]["gain_pct"] > 0
        for p in ("VAL", "HIST")
    )
    account_df = pd.DataFrame(accounts)
    cost_gate = all(
        (
            (account_df["phase"] == p)
            & (account_df["cost_bp"] == 1.0)
            & (account_df["annual_return_linear"] > 0)
        ).any()
        for p in ("VAL", "HIST")
    )
    reconstruct_map = {d["phase"]: d for d in reconstruction}
    not_reconstructable = all(
        np.isfinite(reconstruct_map[p]["hgb_r2"])
        and reconstruct_map[p]["hgb_r2"] < 0.75
        for p in ("VAL", "HIST")
    )
    fresh = direct_map.get("FRESH", {})
    fresh_direction = bool(
        fresh.get("dates", 0) >= 25
        and np.isfinite(fresh.get("mean_product_tsic", np.nan))
        and fresh.get("mean_product_tsic", 0) > 0
    )
    historical_admit = bool(
        row["selection_note"] == "passed_development_gate"
        and point_positive
        and direction_positive
        and heldout_positive
        and cost_gate
        and not_reconstructable
        and multiplicity.get("family_maxT_p", 1.0) <= 0.10
    )
    production_admit = bool(historical_admit and interval_positive and fresh_direction)

    result = {
        "candidate": candidate,
        "mechanism": row["mechanism"],
        "formula_role": row["formula_role"],
        "span": int(row["span"]),
        "horizon": h,
        "orientation": orientation,
        "selection_note": row["selection_note"],
        "development": {
            "dev_a_tsic": float(row["dev_a_tsic"]),
            "dev_b1_tsic": float(row["dev_b1_tsic"]),
            "dev_b2_tsic": float(row["dev_b2_tsic"]),
            "dev_fold1_ridge_gain_pct": float(row["dev_fold1_ridge_gain_pct"]),
            "dev_fold2_ridge_gain_pct": float(row["dev_fold2_ridge_gain_pct"]),
        },
        "direct": direct,
        "model_increment": model_rows,
        "reconstruction": reconstruction,
        "heldout_asset": heldout,
        "accounts": accounts,
        "multiplicity": multiplicity,
        "gates": {
            "all_val_hist_model_points_positive": point_positive,
            "val_hist_ridge_norm_block_low_positive": interval_positive,
            "val_hist_direct_direction_positive": direction_positive,
            "val_hist_heldout_positive": heldout_positive,
            "val_hist_1bp_account_positive": cost_gate,
            "val_hist_not_reconstructable_hgb": not_reconstructable,
            "fresh_direction_positive_with_25_dates": fresh_direction,
        },
        "historical_default_off_factor": historical_admit,
        "production_independent_alpha": production_admit,
        "prediction_files": prediction_paths,
    }
    frame.drop(columns=[oriented], inplace=True)
    return result


def attempt_hourly_analogue(daily_winner: dict) -> dict:
    # This is intentionally a bounded diagnostic, not a second discovery search.
    ticker_subset = [t for t in TICKERS if t in {"ES=F", "NQ=F", "RTY=F", "ZN=F", "ZB=F", "GC=F", "SI=F", "HG=F", "CL=F", "NG=F", "ZC=F", "ZW=F", "ZS=F", "6E=F", "6J=F"}]
    frames = []
    for ticker in ticker_subset:
        try:
            data = yf.download(
                ticker,
                period="700d",
                interval="60m",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
                timeout=30,
            )
            data = _flatten_columns(data).reset_index()
            time_col = "datetime" if "datetime" in data.columns else data.columns[0]
            data = data.rename(columns={time_col: "date"})
            data["date"] = pd.to_datetime(data["date"], utc=True).dt.tz_convert(None)
            data["ticker"] = ticker
            data["sector"] = TICKERS[ticker]
            data = data.dropna(subset=["open", "high", "low", "close"])
            if len(data) >= 1000:
                frames.append(data[["date", "ticker", "sector", "open", "high", "low", "close", "volume"]])
        except Exception:
            continue
    if len(frames) < 8:
        return {"status": "not_run_insufficient_hourly_assets", "assets": len(frames)}
    panel = pd.concat(frames, ignore_index=True)
    # Reuse the same completed-bar state logic at bar frequency. Calendar-year
    # phase labels are not reused; split the last 35% chronologically.
    built, metas, parents = build_features(panel)
    name = daily_winner["candidate"]
    if name not in built.columns:
        return {"status": "not_run_candidate_unavailable", "assets": len(frames)}
    orientation = float(daily_winner["orientation"])
    built["hourly_signal"] = orientation * built[name]
    dates = sorted(built["date"].dropna().unique())
    split = dates[int(len(dates) * 0.65)]
    test = built[built["date"] >= split]
    metrics = {}
    for h in (1, 4):
        target = f"target_norm_h{h}"
        if target not in test:
            continue
        product_ics = []
        for _, g in test.groupby("ticker"):
            ic = safe_spearman(g["hourly_signal"], g[target], min_n=150)
            if np.isfinite(ic):
                product_ics.append(ic)
        metrics[f"h{h}"] = {
            "mean_product_tsic": float(np.mean(product_ics)) if product_ics else np.nan,
            "positive_share": float(np.mean(np.asarray(product_ics) > 0)) if product_ics else np.nan,
            "products": int(len(product_ics)),
        }
    return {
        "status": "bounded_hourly_homologue_completed",
        "assets": len(frames),
        "split": str(pd.Timestamp(split)),
        "metrics": metrics,
        "admitted": False,
        "reason": "diagnostic homologue only; no independent selection or full cost model",
    }


def write_markdown(report: dict) -> None:
    lines = [
        "# Quantum open-system factor search",
        "",
        "This public-runner study is a transportability experiment on continuous global-futures proxies. It does not certify a production roll implementation or literal quantum market mechanism.",
        "",
        f"- generated: `{report['generated_at']}`",
        f"- accepted assets: **{report['data']['accepted_assets']}**",
        f"- observations: **{report['data']['rows']}**",
        f"- date range: **{report['data']['start']} — {report['data']['end']}**",
        f"- candidate declarations: **{report['candidate_family_size']}**",
        f"- finalists: **{len(report['finalists'])}**",
        "",
        "## Decisions",
        "",
    ]
    for item in report["finalists"]:
        lines.extend([
            f"### `{item['candidate']}` — h={item['horizon']}",
            "",
            f"- mechanism: `{item['mechanism']}`",
            f"- historical default-off factor: **{item['historical_default_off_factor']}**",
            f"- production independent alpha: **{item['production_independent_alpha']}**",
            f"- family max-T p: `{item['multiplicity'].get('family_maxT_p')}`",
            "- gates: " + ", ".join(f"{k}={v}" for k, v in item["gates"].items()),
            "",
            "| phase | TSIC | CSIC | positive products |",
            "|---|---:|---:|---:|",
        ])
        for d in item["direct"]:
            lines.append(
                f"| {d['phase']} | {d['mean_product_tsic']:.6f} | {d['mean_daily_csic']:.6f} | {d['positive_product_share']:.3f} |"
            )
        lines.extend(["", "| target | model | phase | MSE gain % | 95% low | 95% high |", "|---|---|---|---:|---:|---:|"])
        for m in item["model_increment"]:
            if m["phase"] in {"VAL", "HIST", "LATE", "FRESH"}:
                lines.append(
                    f"| {m['target_kind']} | {m['model_type']} | {m['phase']} | {m['gain_pct']:.6f} | {m['lo']:.6f} | {m['hi']:.6f} |"
                )
        lines.append("")
    lines.extend([
        "## Governance boundary",
        "",
        "Candidates were oriented and selected only on the development periods. Later periods were not used to redefine formulas. The post-2026-09-11 segment is reported separately. Any admitted object remains default-off unless the fresh interval and execution evidence are sufficient.",
        "",
        "The continuous futures proxies can contain vendor roll conventions. All production claims require replay on the canonical point-in-time roll-safe Chinese-futures panel.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    panel, coverage = download_panel()
    frame, candidates, parent_cols = build_features(panel)
    # Remove impossible return spikes from target/scoring rows while preserving
    # the state-history clock. This is a diagnostic vendor-roll guard.
    for h in HORIZONS:
        raw = frame[f"target_raw_h{h}"]
        invalid = raw.abs() > (0.35 * math.sqrt(h))
        frame.loc[invalid, [f"target_raw_h{h}", f"target_norm_h{h}", f"target_open_h{h}"]] = np.nan

    screen = screen_candidates(frame, candidates, parent_cols)
    finalists = select_finalists(screen)
    results = []
    for _, row in finalists.iterrows():
        results.append(evaluate_finalist(frame, screen, row, parent_cols))

    historical = [x for x in results if x["historical_default_off_factor"]]
    winner = None
    if historical:
        def score(x: dict) -> float:
            vals = [
                m["gain_pct"] for m in x["model_increment"]
                if m["phase"] in {"VAL", "HIST"}
                and m["target_kind"] == "norm"
                and m["model_type"] == "ridge"
                and np.isfinite(m["gain_pct"])
            ]
            return min(vals) if vals else -np.inf
        winner = max(historical, key=score)
    hourly = attempt_hourly_analogue(winner) if winner is not None else {"status": "not_run_no_daily_admission"}

    report = {
        "schema_version": 1,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "study": "quantum_open_system_measurement_pointer_search",
        "repository_target": "ZZZZYANWANG/Trade",
        "data": {
            "provider": "Yahoo Finance via yfinance",
            "continuous_futures_proxy": True,
            "accepted_assets": int(coverage["accepted"].sum()),
            "rows": int(len(frame)),
            "start": str(frame["date"].min().date()),
            "end": str(frame["date"].max().date()),
            "fresh_cutoff": "2026-09-11",
        },
        "candidate_family_size": int(len(candidates) * len(HORIZONS)),
        "parent_feature_count": int(len(parent_cols)),
        "selection_protocol": {
            "formula_orientation": "DEV_A only",
            "development_replication": ["DEV_B1", "DEV_B2"],
            "outer_periods": ["VAL", "HIST", "LATE", "FRESH"],
            "one_per_horizon_mechanism": True,
            "raw_quantum_coordinates_in_parent_set": True,
            "models": ["Ridge", "HistGradientBoosting"],
            "heldout_assets": "three deterministic product folds",
            "multiplicity": "moving-block family max-T on daily cross-sectional IC",
        },
        "finalists": results,
        "hourly_homologue": hourly,
        "admitted_historical_default_off": [x["candidate"] for x in historical],
        "admitted_production_independent_alpha": [x["candidate"] for x in results if x["production_independent_alpha"]],
    }
    (OUT / "RESULTS.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_markdown(report)
    frame[["date", "ticker", "sector"] + parent_cols + [m.candidate for m in candidates] + [
        c for h in HORIZONS for c in (f"target_raw_h{h}", f"target_norm_h{h}", f"target_open_h{h}")
    ]].to_parquet(OUT / "research_panel.parquet", index=False)
    print(json.dumps({
        "accepted_assets": report["data"]["accepted_assets"],
        "rows": report["data"]["rows"],
        "finalists": len(results),
        "historical_admissions": report["admitted_historical_default_off"],
        "production_admissions": report["admitted_production_independent_alpha"],
        "hourly_status": hourly.get("status"),
    }, indent=2))


if __name__ == "__main__":
    main()
