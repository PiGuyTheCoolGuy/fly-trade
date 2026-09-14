"""Causal features with a finite window shared by training and live inference."""

import numpy as np
import pandas as pd

WARMUP = 96
FEATURE_VERSION = 1


def features(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame.close
    ret = np.log(close).diff()
    result = {}
    for lag in (1, 3, 6, 12, 24):
        result[f"return_{lag}"] = np.log(close / close.shift(lag))
    for window in (6, 24, 96):
        result[f"volatility_{window}"] = ret.rolling(window).std()
        result[f"trend_{window}"] = close / close.rolling(window).mean() - 1
    change = close.diff()
    gain = change.clip(lower=0).rolling(14).mean()
    loss = (-change.clip(upper=0)).rolling(14).mean()
    result["rsi"] = (gain / (gain + loss).replace(0, np.nan)).fillna(0.5) - 0.5
    result["range"] = (frame.high - frame.low) / close
    result["body"] = (close - frame.open) / frame.open
    log_volume = np.log1p(frame.volume)
    result["volume_z"] = ((log_volume - log_volume.rolling(24).mean())
                          / log_volume.rolling(24).std().replace(0, np.nan)).fillna(0)
    seconds = frame.index.to_numpy() % 86400
    result["hour_sin"] = np.sin(seconds * (2 * np.pi / 86400))
    result["hour_cos"] = np.cos(seconds * (2 * np.pi / 86400))
    output = pd.DataFrame(result, index=frame.index).replace([np.inf, -np.inf], np.nan)
    output.iloc[:WARMUP] = np.nan
    return output


def targets(frame: pd.DataFrame, horizon: int) -> pd.Series:
    # Observe close[t], buy at open[t+1], label uses a later open.
    return np.log(frame.open.shift(-(horizon + 1)) / frame.open.shift(-1))
