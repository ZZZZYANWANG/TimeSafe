from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
from pathlib import Path

import baostock as bs
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

OUT = Path("research/fluid_round19/results")
OUT.mkdir(parents=True, exist_ok=True)

START = "2018-01-01"
END = "2026-11-18"
N_CODES = 120
MIN_GOOD_CODES = 60
TOP_LIQUID = 80
HORIZONS = [1, 5, 10, 20]
LAGS = [1, 2, 3, 5, 10, 20, 40, 60]
EPS = 1e-12
RNG_SEED = 20261119

DEV_START, DEV_END = pd.Timestamp("2019-01-01"), pd.Timestamp("2021-12-31")
VAL_START, VAL_END = pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31")
HOLD_START, HOLD_END = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")
SHADOW_START, SHADOW_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-11-18")

FAMILY_COLS = {
    "source_characteristic_residual": [
        "mz_source_characteristic_residual",
        "mz_source_characteristic_residual_ewm5",
        "mz_source_characteristic_residual_innov60",
    ],
    "kh_source_loading": [
        "mz_kh_source_loading",
        "mz_kh_source_loading_ewm5",
        "mz_kh_source_loading_innov60",
    ],
    "impedance_pressure_release": [
        "mz_impedance_pressure_release",
        "mz_impedance_pressure_release_ewm5",
        "mz_impedance_pressure_release_innov60",
    ],
    "memory_centroid_momentum": [
        "mz_memory_centroid_momentum",
        "mz_memory_centroid_momentum_ewm5",
        "mz_memory_centroid_momentum_innov60",
    ],
    "unresolved_variance_source_convexity": [
        "mz_unresolved_variance_source_convexity",
        "mz_unresolved_variance_source_convexity_ewm5",
        "mz_unresolved_variance_source_convexity_innov60",
    ],
    "constitutive_recovery_violation": [
        "mz_constitutive_recovery_violation",
        "mz_constitutive_recovery_violation_ewm5",
        "mz_constitutive_recovery_violation_innov60",
    ],
    "lagrangian_source_memory_transport": [
        "mz_lagrangian_source_memory_transport",
        "mz_lagrangian_source_memory_transport_ewm5",
        "mz_lagrangian_source_memory_transport_innov60",
    ],
}

PARENT_COLS = [
    "mom1", "mom5", "mom10", "mom20", "mom60",
    "vol5", "vol20", "vol60", "range1", "range20",
    "amount_z20", "amount_chg1", "overnight", "intraday_ret",
    "rv_intraday", "path_efficiency", "u_early", "u_late", "shear",
    "rho_late", "pressure_gap", "reflection", "lmfl4",
    "variance_drag", "source_raw",
]


def frame_from_result(rs) -> pd.DataFrame:
    if rs.error_code != "0":
        raise RuntimeError(f"baostock error {rs.error_code}: {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def numericize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_universe() -> tuple[list[str], str]:
    snapshots = ["2021-12-31", "2021-12-30", "2021-12-29", "2021-09-30", ""]
    chosen = None
    chosen_date = None
    for date in snapshots:
        try:
            rs = bs.query_hs300_stocks(date=date)
            df = frame_from_result(rs)
        except Exception:
            continue
        if len(df) >= N_CODES:
            chosen = sorted(df["code"].dropna().unique().tolist())
            chosen_date = date or "current_fallback"
            break
    if chosen is None:
        raise RuntimeError("could not obtain HS300 universe")
    idx = np.linspace(0, len(chosen) - 1, N_CODES, dtype=int)
    codes = list(dict.fromkeys(chosen[i] for i in idx))
    if len(codes) < N_CODES:
        for code in chosen:
            if code not in codes:
                codes.append(code)
            if len(codes) >= N_CODES:
                break
    return codes[:N_CODES], str(chosen_date)


def query_history(code: str, frequency: str) -> pd.DataFrame:
    if frequency == "d":
        fields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,pctChg,tradestatus,isST"
    else:
        fields = "date,time,code,open,high,low,close,volume,amount,adjustflag"
    last_error = None
    for attempt in range(3):
        try:
            rs = bs.query_history_k_data_plus(
                code,
                fields,
                start_date=START,
                end_date=END,
                frequency=frequency,
                adjustflag="2",
            )
            df = frame_from_result(rs)
            if not df.empty:
                return df
        except Exception as exc:
            last_error = exc
        time.sleep(1.0 + attempt)
    raise RuntimeError(f"history failed for {code} {frequency}: {last_error}")


def fetch_data(codes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_parts = []
    minute_parts = []
    log_rows = []
    for i, code in enumerate(codes, 1):
        status = {"code": code, "daily_rows": 0, "minute_rows": 0, "ok": False, "error": ""}
        try:
            d = query_history(code, "d")
            m = query_history(code, "60")
            status["daily_rows"] = int(len(d))
            status["minute_rows"] = int(len(m))
            if len(d) < 500 or len(m) < 800:
                raise RuntimeError("insufficient history")
            daily_parts.append(d)
            minute_parts.append(m)
            status["ok"] = True
        except Exception as exc:
            status["error"] = repr(exc)
        log_rows.append(status)
        pd.DataFrame(log_rows).to_csv(OUT / "fetch_log.csv", index=False)
        print(f"FETCH {i}/{len(codes)} {code} ok={status['ok']} d={status['daily_rows']} m={status['minute_rows']}", flush=True)
        time.sleep(0.05)
    if len(daily_parts) < MIN_GOOD_CODES:
        raise RuntimeError(f"only {len(daily_parts)} usable codes")
    daily = pd.concat(daily_parts, ignore_index=True)
    minute = pd.concat(minute_parts, ignore_index=True)
    fetch_log = pd.DataFrame(log_rows)
    return daily, minute, fetch_log


def aggregate_intraday(minute: pd.DataFrame) -> pd.DataFrame:
    minute = minute.copy()
    minute["date"] = pd.to_datetime(minute["date"], errors="coerce")
    minute["time"] = minute["time"].astype(str)
    minute["ts"] = pd.to_datetime(minute["time"].str.slice(0, 14), format="%Y%m%d%H%M%S", errors="coerce")
    minute = numericize(minute, ["open", "high", "low", "close", "volume", "amount"])
    minute = minute.dropna(subset=["date", "code", "open", "close", "ts"])
    rows = []
    grouped = minute.groupby(["code", "date"], sort=False)
    for (code, date), g in grouped:
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < 4:
            continue
        chunks = np.array_split(np.arange(len(g)), 4)
        bars = []
        for idx in chunks:
            q = g.iloc[idx]
            bars.append({
                "open": float(q["open"].iloc[0]),
                "high": float(q["high"].max()),
                "low": float(q["low"].min()),
                "close": float(q["close"].iloc[-1]),
                "volume": float(q["volume"].sum()),
                "amount": float(q["amount"].sum()),
            })
        total_amount = sum(max(b["amount"], 0.0) for b in bars)
        total_volume = sum(max(b["volume"], 0.0) for b in bars)
        if total_amount <= 0 or total_volume <= 0:
            continue
        rec = {"code": code, "date": date, "minute_bars": int(len(g))}
        prev_close = None
        for k, b in enumerate(bars, 1):
            denom = b["open"] if prev_close is None else prev_close
            rec[f"seg_r{k}"] = math.log(max(b["close"], EPS) / max(denom, EPS))
            rec[f"seg_amount{k}"] = b["amount"]
            rec[f"seg_volume{k}"] = b["volume"]
            rec[f"seg_ashare{k}"] = b["amount"] / total_amount
            rec[f"seg_vshare{k}"] = b["volume"] / total_volume
            prev_close = b["close"]
        rec["path_open"] = bars[0]["open"]
        rec["path_close"] = bars[-1]["close"]
        rows.append(rec)
    return pd.DataFrame(rows)


def prior_z(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    mp = min_periods or max(20, window // 2)
    mu = s.rolling(window, min_periods=mp).mean().shift(1)
    sd = s.rolling(window, min_periods=mp).std(ddof=0).shift(1)
    return (s - mu) / (sd + EPS)


def rolling_slope(y: pd.Series, x: pd.Series, window: int = 60) -> pd.Series:
    cov = y.rolling(window, min_periods=30).cov(x).shift(1)
    var = x.rolling(window, min_periods=30).var(ddof=0).shift(1)
    return cov / (var + EPS)


def causal_memory(x: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    num = pd.Series(0.0, index=x.index)
    den = pd.Series(0.0, index=x.index)
    cent = pd.Series(0.0, index=x.index)
    for lag in LAGS:
        corr = x.rolling(252, min_periods=126).corr(x.shift(lag)).shift(1).clip(-0.95, 0.95)
        w = corr.abs()
        num = num.add(corr * x.shift(lag), fill_value=0.0)
        den = den.add(w, fill_value=0.0)
        cent = cent.add(w * lag, fill_value=0.0)
    mem = num / den.replace(0.0, np.nan)
    centroid = cent / den.replace(0.0, np.nan)
    coherence = den / len(LAGS)
    return mem, centroid, coherence


def row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    am = a - np.nanmean(a, axis=1, keepdims=True)
    bm = b - np.nanmean(b, axis=1, keepdims=True)
    den = np.sqrt(np.nansum(am * am, axis=1) * np.nansum(bm * bm, axis=1))
    out = np.nansum(am * bm, axis=1) / np.where(den > EPS, den, np.nan)
    return out


def enrich_code(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    g["ret1"] = np.log(g["close"] / g["close"].shift(1))
    for h in [1, 5, 10, 20, 60]:
        g[f"mom{h}"] = np.log(g["close"] / g["close"].shift(h))
    for w in [5, 20, 60]:
        g[f"vol{w}"] = g["ret1"].rolling(w, min_periods=max(3, w // 2)).std(ddof=0).shift(1)
    g["range1"] = np.log(g["high"] / g["low"])
    g["range20"] = g["range1"].rolling(20, min_periods=10).mean().shift(1)
    g["log_amount"] = np.log1p(g["amount"].clip(lower=0.0))
    g["log_volume"] = np.log1p(g["volume"].clip(lower=0.0))
    g["amount_z20"] = prior_z(g["log_amount"], 20, 10)
    g["amount_chg1"] = g["log_amount"].diff()
    g["overnight"] = np.log(g["open"] / g["preclose"])
    g["intraday_ret"] = np.log(g["close"] / g["open"])
    sigma = g["ret1"].rolling(20, min_periods=15).std(ddof=0).shift(1).clip(lower=1e-4)
    for k in range(1, 5):
        g[f"u{k}"] = g[f"seg_r{k}"] / sigma
        g[f"rho{k}"] = g[f"seg_ashare{k}"].clip(lower=1e-5, upper=1.0)
        g[f"p{k}"] = g[f"u{k}"] * g[f"u{k}"].abs() / (g[f"rho{k}"] + 0.02)
    u = np.column_stack([g[f"u{k}"].to_numpy(float) for k in range(1, 5)])
    rho = np.column_stack([g[f"rho{k}"].to_numpy(float) for k in range(1, 5)])
    g["rv_intraday"] = np.nansum(u * u, axis=1)
    g["path_efficiency"] = np.abs(np.nansum(u, axis=1)) / (np.nansum(np.abs(u), axis=1) + EPS)
    g["u_early"] = g["u1"] + g["u2"]
    g["u_late"] = g["u3"] + g["u4"]
    g["shear"] = g["u_late"] - g["u_early"]
    g["rho_early"] = g["rho1"] + g["rho2"]
    g["rho_late"] = g["rho3"] + g["rho4"]
    g["p_early"] = g["p1"] + g["p2"]
    g["p_late"] = g["p3"] + g["p4"]
    g["pressure_gap"] = g["p_late"] - g["p_early"]
    log_rho_e = np.log(g["rho_early"] + EPS)
    log_rho_l = np.log(g["rho_late"] + EPS)
    k_e = rolling_slope(g["p_early"], log_rho_e, 60)
    k_l = rolling_slope(g["p_late"], log_rho_l, 60)
    c_e = np.sqrt((k_e.abs() / (g["rho_early"] + EPS)).clip(lower=0.0)).clip(0.0, 20.0)
    c_l = np.sqrt((k_l.abs() / (g["rho_late"] + EPS)).clip(lower=0.0)).clip(0.0, 20.0)
    z_e = g["rho_early"] * c_e
    z_l = g["rho_late"] * c_l
    g["reflection"] = (z_l - z_e) / (z_l.abs() + z_e.abs() + EPS)
    j_e = g["u_early"] + g["p_early"] / (z_e + 0.05)
    j_l = g["u_late"] + g["p_late"] / (z_l + 0.05)
    lambda_char = 0.5 * (g["u_early"] + g["u_late"] + c_e + c_l)
    source_raw = (
        prior_z(g["log_amount"].diff(), 60, 30)
        + prior_z(g["rho_late"].diff(), 60, 30)
        + 0.5 * prior_z(g["log_volume"].diff(), 60, 30)
    )
    g["source_raw"] = source_raw
    mem_source, cent_source, coh_source = causal_memory(source_raw)
    mem_pressure, cent_pressure, coh_pressure = causal_memory(g["pressure_gap"])
    mem_j, cent_j, coh_j = causal_memory(j_l)
    density_grad = np.log((g["rho_late"] + EPS) / (g["rho_early"] + EPS))
    ri = (-density_grad * g["pressure_gap"] / (g["shear"] * g["shear"] + 0.05)).clip(-10.0, 10.0)
    gamma = g["shear"].abs() * np.sqrt((1.0 - 4.0 * ri).clip(lower=0.0, upper=25.0))
    temporal_j = j_l.diff()
    spatial_j = j_l - j_e
    scr = temporal_j + lambda_char * spatial_j - source_raw - mem_source
    g["mz_source_characteristic_residual"] = -prior_z(scr, 120, 60)
    g["mz_kh_source_loading"] = -gamma * (source_raw + mem_source) * np.exp(-g["pressure_gap"].abs().clip(upper=10.0) / 3.0)
    g["mz_impedance_pressure_release"] = -g["reflection"] * (g["pressure_gap"] + mem_pressure)
    g["mz_memory_centroid_momentum"] = prior_z(cent_pressure, 120, 60) * prior_z(g["mom20"], 120, 60)
    m = np.abs(u) / (np.sqrt(rho) + 0.05)
    mbar = np.nansum(rho * m, axis=1)
    var_drag = np.nansum(rho * (m - mbar[:, None]) ** 2, axis=1)
    convex = np.nansum(rho / (1.0 + m), axis=1) - 1.0 / (1.0 + mbar)
    loading = np.nansum(rho * u * u, axis=1)
    g["variance_drag"] = var_drag
    g["mz_unresolved_variance_source_convexity"] = -var_drag + loading * convex
    load = u * u
    front_loaded = (np.nansum(load[:, :2], axis=1) > np.nansum(load[:, 2:], axis=1)).astype(float)
    late_reacc = np.maximum(m[:, 2] - m[:, 1], 0.0) + np.maximum(m[:, 3] - m[:, 2], 0.0)
    direction = np.sign(np.nansum(u, axis=1))
    g["mz_constitutive_recovery_violation"] = -direction * front_loaded * late_reacc
    g["lmfl4"] = row_corr(rho[:, :3], u[:, 1:])
    g["mz_lagrangian_source_memory_transport"] = g["lmfl4"] * (1.0 + prior_z(cent_source, 120, 60) / 3.0) * (1.0 + coh_source.clip(0.0, 2.0) / 4.0)
    raw_cols = [cols[0] for cols in FAMILY_COLS.values()]
    for col in raw_cols:
        g[col + "_ewm5"] = g[col].ewm(span=5, adjust=False, min_periods=3).mean()
        g[col + "_innov60"] = prior_z(g[col], 60, 30)
    return g


def prepare_panel(daily: pd.DataFrame, minute: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = numericize(daily, ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg", "tradestatus", "isST"])
    daily = daily.dropna(subset=["date", "code", "open", "high", "low", "close"])
    daily = daily.drop_duplicates(["code", "date"], keep="last")
    path = aggregate_intraday(minute)
    panel = daily.merge(path, on=["code", "date"], how="inner", validate="one_to_one")
    panel = panel.sort_values(["code", "date"])
    enriched = [enrich_code(g) for _, g in panel.groupby("code", sort=False)]
    panel = pd.concat(enriched, ignore_index=True)
    panel["liq_ma20"] = panel.groupby("code")["log_amount"].transform(lambda s: s.rolling(20, min_periods=10).mean().shift(1))
    panel["liq_rank"] = panel.groupby("date")["liq_ma20"].rank(method="first", ascending=False)
    panel["eligible"] = (
        (panel["liq_rank"] <= TOP_LIQUID)
        & (panel["tradestatus"].fillna(0) == 1)
        & (panel["isST"].fillna(1) == 0)
        & (panel["minute_bars"] >= 4)
    )
    panel["pool"] = np.where(panel["liq_rank"] <= TOP_LIQUID // 2, "high", "secondary")
    open_p = daily.pivot(index="date", columns="code", values="open").sort_index()
    close_p = daily.pivot(index="date", columns="code", values="close").reindex(open_p.index)
    targets = []
    for h in HORIZONS:
        t = np.log(close_p.shift(-h) / open_p.shift(-1))
        s = t.stack(dropna=False).rename(f"target_{h}d").reset_index()
        targets.append(s)
    for t in targets:
        panel = panel.merge(t, on=["date", "code"], how="left")
    panel = panel[panel["eligible"]].copy()
    audit = {
        "rows": int(len(panel)),
        "codes": int(panel["code"].nunique()),
        "dates": int(panel["date"].nunique()),
        "start": str(panel["date"].min().date()),
        "end": str(panel["date"].max().date()),
        "path_rows": int(len(path)),
        "candidate_columns": int(sum(len(x) for x in FAMILY_COLS.values())),
        "parent_columns": int(len(PARENT_COLS)),
    }
    return panel, audit


def period_mask(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return (df["date"] >= start) & (df["date"] <= end)


def daily_ic_series(df: pd.DataFrame, signal: pd.Series, target: str, min_n: int = 20) -> pd.Series:
    x = pd.DataFrame({"date": df["date"].to_numpy(), "x": signal.to_numpy(float), "y": df[target].to_numpy(float)})
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    vals = {}
    for date, g in x.groupby("date", sort=True):
        if len(g) < min_n or g["x"].nunique() < 3 or g["y"].nunique() < 3:
            continue
        vals[pd.Timestamp(date)] = float(g["x"].rank().corr(g["y"].rank()))
    return pd.Series(vals, dtype=float).sort_index()


def block_bootstrap(series: pd.Series, reps: int = 2000, block: int = 20) -> dict:
    v = series.dropna().to_numpy(float)
    n = len(v)
    if n < 40:
        return {"n": n, "mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_nonpositive": 1.0}
    rng = np.random.default_rng(RNG_SEED + n)
    means = np.empty(reps)
    need = int(math.ceil(n / block))
    offsets = np.arange(block)
    for b in range(reps):
        starts = rng.integers(0, n, size=need)
        idx = np.concatenate([(s + offsets) % n for s in starts])[:n]
        means[b] = np.mean(v[idx])
    return {
        "n": n,
        "mean": float(np.mean(v)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_nonpositive": float((1 + np.sum(means <= 0.0)) / (reps + 1)),
    }


def holm_adjust(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    prev = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        prev = max(prev, adj)
        out[name] = prev
    return out


def synchronized_max_t(ic_map: dict[str, pd.Series], reps: int = 2000, block: int = 20) -> dict[str, float]:
    mat = pd.concat(ic_map, axis=1).dropna()
    if len(mat) < 80:
        return {k: 1.0 for k in ic_map}
    x = mat.to_numpy(float)
    n, m = x.shape
    obs = x.mean(axis=0) / (x.std(axis=0, ddof=1) / math.sqrt(n) + EPS)
    centered = x - x.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(RNG_SEED + 101)
    need = int(math.ceil(n / block))
    offsets = np.arange(block)
    maxima = np.empty(reps)
    for b in range(reps):
        starts = rng.integers(0, n, size=need)
        idx = np.concatenate([(s + offsets) % n for s in starts])[:n]
        q = centered[idx]
        t = q.mean(axis=0) / (q.std(axis=0, ddof=1) / math.sqrt(n) + EPS)
        maxima[b] = np.nanmax(t)
    return {name: float((1 + np.sum(maxima >= obs[j])) / (reps + 1)) for j, name in enumerate(mat.columns)}


def model_increment(train: pd.DataFrame, test: pd.DataFrame, factor: str, direction: int, target: str) -> dict:
    cols = [c for c in PARENT_COLS if c in train]
    tr = train[cols + [factor, target]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    te = test[cols + [factor, target, "date"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(tr) < 3000 or len(te) < 1000:
        return {"n_train": len(tr), "n_test": len(te), "mse_improvement": np.nan, "residual_rankic": np.nan, "max_parent_corr": np.nan}
    tr[factor] *= direction
    te[factor] *= direction
    base = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    full = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    base.fit(tr[cols], tr[target])
    full.fit(tr[cols + [factor]], tr[target])
    y = te[target].to_numpy(float)
    pb = base.predict(te[cols])
    pf = full.predict(te[cols + [factor]])
    mse_b = float(np.mean((y - pb) ** 2))
    mse_f = float(np.mean((y - pf) ** 2))
    recon = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    recon.fit(tr[cols], tr[factor])
    residual = pd.Series(te[factor].to_numpy(float) - recon.predict(te[cols]), index=te.index)
    ric = daily_ic_series(te, residual, target)
    parent_corrs = []
    for col in cols:
        s = daily_ic_series(te, te[factor], col)
        if len(s):
            parent_corrs.append(abs(float(s.mean())))
    return {
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "mse_improvement": float((mse_b - mse_f) / (mse_b + EPS)),
        "residual_rankic": float(ric.mean()) if len(ric) else np.nan,
        "max_parent_corr": float(max(parent_corrs)) if parent_corrs else np.nan,
    }


def portfolio_cost(df: pd.DataFrame, factor: str, direction: int, target: str, horizon: int, cost_bp: float = 5.0) -> dict:
    z = df[["date", "code", factor, target]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    z["signal"] = z[factor] * direction
    dates = sorted(z["date"].unique())
    offset_stats = []
    for offset in range(max(1, horizon)):
        chosen_dates = dates[offset::max(1, horizon)]
        old = {}
        returns = []
        turnovers = []
        for date in chosen_dates:
            g = z[z["date"] == date].sort_values("signal")
            if len(g) < 30:
                continue
            q = max(3, int(math.floor(len(g) * 0.10)))
            low = g.head(q)
            high = g.tail(q)
            w = {code: -0.5 / q for code in low["code"]}
            for code in high["code"]:
                w[code] = w.get(code, 0.0) + 0.5 / q
            ymap = dict(zip(g["code"], g[target]))
            gross = sum(weight * float(ymap.get(code, 0.0)) for code, weight in w.items())
            keys = set(old) | set(w)
            turnover = sum(abs(w.get(k, 0.0) - old.get(k, 0.0)) for k in keys)
            net = gross - turnover * cost_bp * 1e-4
            returns.append(net)
            turnovers.append(turnover)
            old = w
        if len(returns) >= 8:
            arr = np.asarray(returns, float)
            ann = float(arr.mean() * 252.0 / max(1, horizon))
            sharpe = float(arr.mean() / (arr.std(ddof=1) + EPS) * math.sqrt(252.0 / max(1, horizon)))
            offset_stats.append((ann, sharpe, float(np.mean(turnovers)), len(arr)))
    if not offset_stats:
        return {"annual_return": np.nan, "sharpe": np.nan, "avg_turnover": np.nan, "offsets": 0}
    return {
        "annual_return": float(np.mean([x[0] for x in offset_stats])),
        "sharpe": float(np.mean([x[1] for x in offset_stats])),
        "avg_turnover": float(np.mean([x[2] for x in offset_stats])),
        "offsets": int(len(offset_stats)),
    }


def scan_development(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = panel[period_mask(panel, DEV_START, DEV_END)].copy()
    rows = []
    selected = []
    for family, cols in FAMILY_COLS.items():
        family_rows = []
        for col in cols:
            for h in HORIZONS:
                target = f"target_{h}d"
                raw_ic = daily_ic_series(dev, dev[col], target)
                if len(raw_ic) < 180:
                    continue
                direction = 1 if raw_ic.mean() >= 0 else -1
                ic = raw_ic * direction
                yearly = ic.groupby(ic.index.year).mean().to_dict()
                pool_scores = {}
                for pool in ["high", "secondary"]:
                    q = dev[dev["pool"] == pool]
                    s = daily_ic_series(q, q[col] * direction, target)
                    pool_scores[pool] = float(s.mean()) if len(s) else np.nan
                yvals = [float(v) for v in yearly.values() if np.isfinite(v)]
                pvals = [float(v) for v in pool_scores.values() if np.isfinite(v)]
                if len(yvals) < 3 or len(pvals) < 2:
                    continue
                score = min(yvals) + 0.25 * float(ic.mean()) + 0.10 * min(pvals)
                row = {
                    "family": family,
                    "factor": col,
                    "horizon": h,
                    "direction": direction,
                    "rankic": float(ic.mean()),
                    "positive_share": float((ic > 0).mean()),
                    "min_year": float(min(yvals)),
                    "high_pool": pool_scores["high"],
                    "secondary_pool": pool_scores["secondary"],
                    "score": float(score),
                    "n_dates": int(len(ic)),
                }
                for year, val in yearly.items():
                    row[f"year_{year}"] = float(val)
                rows.append(row)
                family_rows.append(row)
        if family_rows:
            best = max(family_rows, key=lambda r: r["score"])
            selected.append(best)
    return pd.DataFrame(rows), pd.DataFrame(selected)


def evaluate_phase(panel: pd.DataFrame, selected: pd.DataFrame, train_mask: pd.Series, test_mask: pd.Series, phase: str, include_multiple_testing: bool) -> pd.DataFrame:
    train = panel[train_mask].copy()
    test = panel[test_mask].copy()
    rows = []
    ic_map = {}
    boot_map = {}
    for rec in selected.to_dict("records"):
        factor = rec["factor"]
        direction = int(rec["direction"])
        h = int(rec["horizon"])
        target = f"target_{h}d"
        ic = daily_ic_series(test, test[factor] * direction, target)
        ic_map[rec["family"]] = ic
        boot = block_bootstrap(ic)
        boot_map[rec["family"]] = boot
        model = model_increment(train, test, factor, direction, target)
        cost = portfolio_cost(test, factor, direction, target, h, 5.0)
        yearly = ic.groupby(ic.index.year).mean().to_dict()
        row = {
            "phase": phase,
            "family": rec["family"],
            "factor": factor,
            "horizon": h,
            "direction": direction,
            "rankic": float(ic.mean()) if len(ic) else np.nan,
            "positive_share": float((ic > 0).mean()) if len(ic) else np.nan,
            "n_dates": int(len(ic)),
            "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"],
            "p_nonpositive": boot["p_nonpositive"],
            **model,
            **{f"cost5_{k}": v for k, v in cost.items()},
        }
        for year, val in yearly.items():
            row[f"year_{year}"] = float(val)
        rows.append(row)
    result = pd.DataFrame(rows)
    if include_multiple_testing and len(result):
        holm = holm_adjust({k: v["p_nonpositive"] for k, v in boot_map.items()})
        maxt = synchronized_max_t(ic_map)
        result["holm_p"] = result["family"].map(holm)
        result["max_t_p"] = result["family"].map(maxt)
    else:
        result["holm_p"] = np.nan
        result["max_t_p"] = np.nan
    return result


def apply_gate(result: pd.DataFrame, phase: str) -> pd.DataFrame:
    result = result.copy()
    gates = []
    for row in result.to_dict("records"):
        years = sorted([k for k in row if k.startswith("year_") and pd.notna(row[k])])
        yearly_positive = bool(years) and all(row[k] > 0 for k in years)
        if phase == "validation":
            gate = (
                row["rankic"] > 0.010
                and row["p_nonpositive"] <= 0.10
                and row["holm_p"] <= 0.10
                and row["max_t_p"] <= 0.10
                and row["mse_improvement"] > 0
                and row["residual_rankic"] > 0.005
                and row["max_parent_corr"] < 0.75
                and row["cost5_annual_return"] > 0
                and yearly_positive
            )
        else:
            gate = (
                row["rankic"] > 0
                and row["mse_improvement"] > 0
                and row["residual_rankic"] > 0
                and row["cost5_annual_return"] > 0
                and yearly_positive
            )
        gates.append(bool(gate))
    result["gate_passed"] = gates
    return result


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock login failed {login.error_code}: {login.error_msg}")
    try:
        codes, snapshot = get_universe()
        daily, minute, fetch_log = fetch_data(codes)
    finally:
        bs.logout()
    panel, audit = prepare_panel(daily, minute)
    audit.update({
        "universe_snapshot": snapshot,
        "requested_codes": N_CODES,
        "usable_codes": int(fetch_log["ok"].sum()),
        "dev": [str(DEV_START.date()), str(DEV_END.date())],
        "validation": [str(VAL_START.date()), str(VAL_END.date())],
        "holdout": [str(HOLD_START.date()), str(HOLD_END.date())],
        "shadow": [str(SHADOW_START.date()), str(SHADOW_END.date())],
        "horizons": HORIZONS,
        "lags": LAGS,
        "candidate_families": list(FAMILY_COLS),
    })
    (OUT / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    scan, selected = scan_development(panel)
    scan.to_csv(OUT / "development_scan.csv", index=False)
    selected.to_csv(OUT / "frozen_selection.csv", index=False)
    val = evaluate_phase(
        panel,
        selected,
        period_mask(panel, DEV_START, DEV_END),
        period_mask(panel, VAL_START, VAL_END),
        "validation",
        True,
    )
    val = apply_gate(val, "validation")
    val.to_csv(OUT / "validation_results.csv", index=False)
    val_pass = set(val.loc[val["gate_passed"], "family"])
    selected_hold = selected[selected["family"].isin(val_pass)].copy()
    if len(selected_hold):
        hold = evaluate_phase(
            panel,
            selected_hold,
            period_mask(panel, DEV_START, VAL_END),
            period_mask(panel, HOLD_START, HOLD_END),
            "holdout",
            False,
        )
        hold = apply_gate(hold, "holdout")
    else:
        hold = pd.DataFrame()
    hold.to_csv(OUT / "holdout_results.csv", index=False)
    hold_pass = set(hold.loc[hold.get("gate_passed", pd.Series(dtype=bool)) == True, "family"]) if len(hold) else set()
    selected_shadow = selected[selected["family"].isin(hold_pass)].copy()
    if len(selected_shadow):
        shadow = evaluate_phase(
            panel,
            selected_shadow,
            period_mask(panel, DEV_START, HOLD_END),
            period_mask(panel, SHADOW_START, SHADOW_END),
            "shadow",
            False,
        )
        shadow = apply_gate(shadow, "shadow")
    else:
        shadow = pd.DataFrame()
    shadow.to_csv(OUT / "shadow_results.csv", index=False)
    shadow_pass = set(shadow.loc[shadow.get("gate_passed", pd.Series(dtype=bool)) == True, "family"]) if len(shadow) else set()
    promoted = []
    for family in sorted(shadow_pass):
        rec = selected[selected["family"] == family].iloc[0].to_dict()
        rec["canonical_id"] = f"ashare_{family}_{int(rec['horizon'])}d"
        rec["default_enabled"] = False
        rec["independent_alpha"] = False
        rec["production_certified"] = False
        rec["fresh_forward_required"] = True
        promoted.append(rec)
    (OUT / "promoted.json").write_text(json.dumps(promoted, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "research_completed": True,
        "data_audit": audit,
        "development_tests": int(len(scan)),
        "frozen_identities": int(len(selected)),
        "validation_passed": int(len(val_pass)),
        "holdout_opened": int(len(selected_hold)),
        "holdout_passed": int(len(hold_pass)),
        "shadow_opened": int(len(selected_shadow)),
        "promoted_count": int(len(promoted)),
        "promoted": promoted,
        "governance": "Only candidates passing development-frozen validation, historical holdout, shadow direction, strong-parent increment, residual IC, selection correction and 5bp cost are promoted. All history is historical, not fresh OOS.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# Public A-share non-Markov fluid factor experiment",
        "",
        f"- Usable codes: {audit['usable_codes']}",
        f"- Eligible panel rows: {audit['rows']}",
        f"- Development tests: {len(scan)}",
        f"- Frozen identities: {len(selected)}",
        f"- Validation passed: {len(val_pass)}",
        f"- Holdout passed: {len(hold_pass)}",
        f"- Promoted: {len(promoted)}",
        "",
        "The experiment tests source-coupled characteristic residuals, memory-conditioned Kelvin-Helmholtz loading, impedance pressure release, memory-centroid momentum interaction, unresolved-variance/source-convexity closure, constitutive recovery violations, and Lagrangian source-memory transport.",
        "",
        "All candidate directions, variants and horizons are frozen on 2019-2021. Validation is 2022-2023. Historical holdout is 2024-2025 and is opened only for validation passers. The 2026 shadow is opened only for holdout passers.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    hashes = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file():
            hashes[path.name] = sha256(path)
    (OUT / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
