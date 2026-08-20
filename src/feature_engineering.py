from __future__ import annotations

import numpy as np
import pandas as pd


def _time_to_minutes(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(
        series.astype("string"),
        format="%H:%M",
        errors="coerce",
    )
    return parsed.dt.hour * 60 + parsed.dt.minute


def add_sleep_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer overnight sleep duration from sleep_time and wake_time.
    """
    required = {"sleep_time", "wake_time"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Cannot create sleep_duration. Missing columns: {sorted(missing)}"
        )

    df = df.copy()

    sleep_minutes = _time_to_minutes(df["sleep_time"])
    wake_minutes = _time_to_minutes(df["wake_time"])

    duration = wake_minutes - sleep_minutes
    duration = duration.where(duration >= 0, duration + 24 * 60)

    df["sleep_duration"] = duration / 60

    return df


def add_class_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer class_size = n_male + n_female.
    Also retain female_ratio for traceability to Task 1 analysis.
    """
    required = {"n_male", "n_female"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Cannot create class_size. Missing columns: {sorted(missing)}"
        )

    df = df.copy()

    df["n_male"] = pd.to_numeric(df["n_male"], errors="coerce")
    df["n_female"] = pd.to_numeric(df["n_female"], errors="coerce")

    df["class_size"] = df["n_male"] + df["n_female"]

    df["female_ratio"] = np.where(
        df["class_size"] > 0,
        df["n_female"] / df["class_size"],
        np.nan,
    )

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply model-relevant feature engineering identified in Task 1.
    """
    df = add_sleep_duration(df)
    df = add_class_size(df)
    return df
