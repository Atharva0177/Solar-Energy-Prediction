"""Recursive multi-step forecasting for the API (PRD §33).

The served models are single-step (one 15-min slot). The API turns them
into multi-step forecasters by recursion: predict t+1, append the prediction
to the site's history, predict t+2, …

Weather covariates for future timestamps are carried forward from the last
observation — there is no NWP feed in v1 (recorded in D-019); solar
geometry and calendar features are exact for any future time. Conformal
radii from Phase 11 (mondrian 0.9) attach per-step bounds using the regime
label known at forecast time.

Latency: features that depend only on the timestamp (calendar, carried
weather, solar geometry) are computed ONCE for the whole tail + horizon;
only prediction-dependent features (lags, rolling stats) are refreshed per
step, incrementally for the single new row (see ``_PowerHistory``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.lag import add_lags
from src.features.rolling import add_rolling_features
from src.features.solar import add_solar_position_features
from src.features.temporal import add_temporal_features
from src.features.weather import add_weather_features
from src.models.xgboost_model import prepare_matrix

CADENCE_S = 900
LAG_STEPS = [1, 2, 4, 8, 24, 48, 96]
ROLLING_WINDOWS_S = [3600, 21600, 86400]
TIMEZONE = "Australia/Melbourne"  # D-007
PERSISTENCE_LAG = pd.Timedelta(86400, unit="s")

SOLAR_COLS = ("solar_elevation_deg", "is_daylight", "azimuth_deg",
              "zenith_deg", "day_length_hours")


def build_features(frame: pd.DataFrame, coords: pd.DataFrame,
                   tz: str = TIMEZONE) -> pd.DataFrame:
    """Full Phase 5 feature families on a small frame (one site).

    Batch path — used by tooling/tests as the semantic reference for the
    incremental serving path below.
    """
    out = add_temporal_features(frame)
    lag_specs = {f"power_lag_{s}": pd.Timedelta(s * CADENCE_S, unit="s")
                 for s in LAG_STEPS}
    out = add_lags(out, lag_specs)
    out = add_rolling_features(
        out, windows=[pd.Timedelta(w, unit="s") for w in ROLLING_WINDOWS_S],
        stats=["mean", "std", "min", "max"], min_periods=1)
    out = add_weather_features(out)
    # drop ALL solar-position outputs before recomputing — stale columns
    # would collide on merge (pandas suffixes them _x/_y) and vanish
    out = out.drop(columns=[c for c in SOLAR_COLS if c in out.columns])
    out = add_solar_position_features(out, coords, tz)
    return out


def _static_frame(tail: pd.DataFrame, coords: pd.DataFrame, horizon: int,
                  tz: str = TIMEZONE) -> pd.DataFrame:
    """Tail + ``horizon`` future rows with timestamp-only features done once.

    Future rows carry the last observed weather (identical to carrying it
    forward through the loop — every future row copies the same observation).
    Lags/rolling are NOT computed here: they depend on fed-back predictions.
    """
    tail = tail.sort_values("timestamp").reset_index(drop=True)
    t_last = tail["timestamp"].max()
    last = tail.iloc[-1]
    future = pd.DataFrame({
        "timestamp": [t_last + pd.Timedelta(i * CADENCE_S, unit="s")
                      for i in range(1, horizon + 1)],
        "site_id": last["site_id"], "campus_id": last["campus_id"],
        "power": np.nan,
    })
    for c in ("temperature", "apparent_temperature", "dew_point_temperature",
              "humidity", "wind_speed", "wind_direction"):
        if c in tail.columns:
            future[c] = last[c]
    frame = pd.concat([tail, future], ignore_index=True)
    frame = add_temporal_features(frame)
    frame = add_weather_features(frame)
    frame = frame.drop(columns=[c for c in SOLAR_COLS if c in frame.columns])
    return add_solar_position_features(frame, coords, tz)


class _PowerHistory:
    """Timestamp-keyed power series with single-row lag/rolling queries.

    Serves exactly the semantics of ``add_lags`` (calendar-exact lookup by
    timestamp, NaN when the prior slot is absent) and ``add_rolling_features``
    (time-based window, ``closed='left'`` → [t−W, t), NaN values contribute
    nothing, pandas ddof=1 std) — verified against the batch functions in
    tests — at microsecond cost instead of a whole-frame rebuild per step.
    """

    def __init__(self, ts: np.ndarray, powers: np.ndarray):
        self.ts = np.asarray(ts, dtype="datetime64[ns]")  # sorted, unique
        self.powers = np.asarray(powers, dtype=float)

    def lag(self, t: pd.Timestamp, delta: pd.Timedelta) -> float:
        prior = np.datetime64(t - delta)
        pos = int(np.searchsorted(self.ts, prior))
        if pos < len(self.ts) and self.ts[pos] == prior:
            return self.powers[pos]  # NaN passes through, like add_lags
        return np.nan

    def roll(self, t: pd.Timestamp, window: pd.Timedelta) -> dict:
        lo = int(np.searchsorted(self.ts, np.datetime64(t - window), side="left"))
        hi = int(np.searchsorted(self.ts, np.datetime64(t), side="left"))
        vals = self.powers[lo:hi]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
        return {"mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if vals.size >= 2 else np.nan,
                "min": float(vals.min()), "max": float(vals.max())}


def recursive_forecast_xgboost(reg, tail: pd.DataFrame, horizon: int,
                               coords: pd.DataFrame,
                               radii: dict | None = None,
                               confidence_level: float = 0.9) -> list[dict]:
    """Feed predictions back as lags; refresh per-row features incrementally."""
    static = _static_frame(tail, coords, horizon)
    hist = _PowerHistory(static["timestamp"].to_numpy(),
                         static["power"].to_numpy())
    n_obs = len(static) - horizon
    roll_windows = [pd.Timedelta(w, unit="s") for w in ROLLING_WINDOWS_S]
    suffixes = [f"{w}s" for w in ROLLING_WINDOWS_S]
    out = []
    for i in range(1, horizon + 1):
        k = n_obs + i - 1
        t = static["timestamp"].iloc[k]
        row = static.iloc[[k]].copy()
        for s in LAG_STEPS:
            row[f"power_lag_{s}"] = hist.lag(t, pd.Timedelta(s * CADENCE_S, unit="s"))
        for window, suf in zip(roll_windows, suffixes):
            r = hist.roll(t, window)
            for stat, val in r.items():
                row[f"power_rolling_{stat}_{suf}"] = val
        pred = float(reg.predict(prepare_matrix(row, reg.feature_cols_))[0])
        hist.powers[k] = pred

        out_row = {"timestamp": t, "prediction": round(pred, 4)}
        if radii is not None:
            day = bool(row["is_daylight"].iloc[0])
            nolag = bool(pd.isna(row["power_lag_1"].iloc[0]))
            label = ("day_" if day else "night_") + ("nolag" if nolag else "lag")
            q = radii["regimes"].get(label, radii["global"])
            out_row.update(lower_bound=round(pred - q, 4),
                           upper_bound=round(pred + q, 4),
                           confidence_level=confidence_level, regime=label)
        out.append(out_row)
    return out


def recursive_forecast_persistence(tail: pd.DataFrame,
                                   horizon: int) -> list[dict]:
    """P(t) = P(t−24h) from history extended with own predictions."""
    series = dict(zip(tail["timestamp"],
                      tail["power"].to_numpy(dtype=float)))
    t_last = tail["timestamp"].max()
    out = []
    for i in range(1, horizon + 1):
        t = t_last + pd.Timedelta(i * CADENCE_S, unit="s")
        val = series.get(t - PERSISTENCE_LAG)
        val = None if val is None or not np.isfinite(val) else round(float(val), 4)
        if val is not None:
            series[t] = float(val)
        out.append({"timestamp": t, "prediction": val})
    return out


def recursive_forecast_sequence(pkg: dict, tail: pd.DataFrame, horizon: int,
                                coords: pd.DataFrame) -> list[dict]:
    """Recursive multi-step for the deep models (Phase 14): lstm/gru/transformer.

    Same append→recurse contract as the XGBoost path, but the per-step feature
    view is the Phase 7 window: the last ``lookback+1`` rows standardized
    with the train-split channel stats, current-step power channels zeroed
    by the same leak guard used in training. Predictions feed back as
    observed history (mask=1) — the accepted error-accumulation compromise
    of D-019's recursion. No conformal radii exist for these models (Phase 11
    calibrated xgboost + persistence only), so bounds are omitted.
    """
    import torch

    from src.models.sequence_model import (
        _mask_current_step,
        build_channel_matrix,
    )

    static = _static_frame(tail, coords, horizon)
    powers = static["power"].to_numpy(dtype=float)
    ts = static["timestamp"].to_numpy()
    need = int(pkg["lookback"]) + 1
    ch_mean = np.asarray(pkg["channel_mean"], dtype=np.float64)
    ch_std = np.asarray(pkg["channel_std"], dtype=np.float64)
    y_mean, y_std = float(pkg["y_mean"]), float(pkg["y_std"])
    model = pkg["model"]
    out = []
    for i in range(1, horizon + 1):
        k = len(ts) - horizon + i - 1
        # window = last `need` rows ending at the current step; only the power
        # column changes between steps (fed-back preds) — timestamp features
        # were computed once in _static_frame
        win = static.iloc[k - need + 1 : k + 1].copy()
        win["power"] = powers[k - need + 1 : k + 1]
        # build_channel_matrix selects the fixed CHANNELS order, matching
        # the exported channel_mean/std layout
        M = build_channel_matrix(win)
        X = torch.from_numpy(
            ((M - ch_mean) / ch_std).astype(np.float32)).unsqueeze(0)
        _mask_current_step(X)
        with torch.no_grad():
            pred = float(model(X)[0]) * y_std + y_mean
        powers[k] = pred
        out.append({"timestamp": pd.Timestamp(ts[k]),
                    "prediction": round(pred, 4)})
    return out
