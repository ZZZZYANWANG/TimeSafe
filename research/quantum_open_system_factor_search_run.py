from __future__ import annotations

import numpy as np
import pandas as pd

import quantum_open_system_factor_search as q
import quantum_open_system_factor_search_fast as f


def corrected_fit_predict_pair(
    train: pd.DataFrame,
    test: pd.DataFrame,
    parent_cols: list[str],
    candidate: str,
    target: str,
    model_type: str,
) -> pd.DataFrame:
    categorical = ["ticker", "sector"] if model_type == "ridge" else []
    tr_cols = list(dict.fromkeys(parent_cols + categorical + [candidate, target]))
    te_cols = list(dict.fromkeys(parent_cols + categorical + [candidate, target, "date", "ticker"]))
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
    if model_type == "hgb" and len(tr) > f.HGB_TRAIN_CAP:
        tr = tr.sample(f.HGB_TRAIN_CAP, random_state=q.SEED).sort_index()
    parent_pipe = f.model_pipeline(model_type, parent_cols, categorical)
    joint_pipe = f.model_pipeline(model_type, parent_cols + [candidate], categorical)
    parent_pipe.fit(tr[parent_cols + categorical], tr[target].astype(float))
    joint_pipe.fit(tr[parent_cols + [candidate] + categorical], tr[target].astype(float))
    out = te[["date", "ticker", target]].copy()
    out["parent_prediction"] = parent_pipe.predict(te[parent_cols + categorical])
    out["joint_prediction"] = joint_pipe.predict(te[parent_cols + [candidate] + categorical])
    out["parent_loss"] = (out[target] - out["parent_prediction"]) ** 2
    out["joint_loss"] = (out[target] - out["joint_prediction"]) ** 2
    return out


f.fit_predict_pair = corrected_fit_predict_pair
q.fit_predict_pair = corrected_fit_predict_pair


if __name__ == "__main__":
    f.main()
