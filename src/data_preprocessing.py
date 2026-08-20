from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def standardise_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise categorical values identified during Task 1 EDA.
    """
    df = df.copy()

    title_case_columns = [
        "CCA",
        "tuition",
        "direct_admission",
        "learning_style",
        "gender",
    ]

    for col in title_case_columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .str.title()
            )

    if "mode_of_transport" in df.columns:
        df["mode_of_transport"] = (
            df["mode_of_transport"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

    if "bag_color" in df.columns:
        df["bag_color"] = (
            df["bag_color"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

    return df


def clean_invalid_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Task 1 established that valid ages are 15 and 16.
    Other values are converted to NaN.
    """
    df = df.copy()

    if "age" not in df.columns:
        return df

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.loc[~df["age"].isin(config.VALID_AGES), "age"] = np.nan

    return df


def _unique_non_null(series: pd.Series) -> list:
    return list(pd.unique(series.dropna()))


def consolidate_duplicate_students(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Consolidate duplicate student records.

    EDA showed duplicate rows often contain complementary missing values
    for attendance_rate and final_test. The available non-null value is kept.

    If a strict field (attendance_rate/final_test) contains multiple
    genuinely different non-null values for one student, raise an error.
    Non-strict disagreements are logged and the first non-null value is kept.
    """
    if config.ID_COLUMN not in df.columns:
        raise ValueError(f"Missing required ID column: {config.ID_COLUMN}")

    if not df[config.ID_COLUMN].duplicated().any():
        empty_conflicts = pd.DataFrame(
            columns=["student_id", "column", "values"]
        )
        return df.copy().reset_index(drop=True), empty_conflicts

    original_columns = list(df.columns)
    clean_records = []
    conflict_records = []

    for student_id, group in df.groupby(
        config.ID_COLUMN,
        sort=False,
        dropna=False,
    ):
        record = {config.ID_COLUMN: student_id}

        for col in original_columns:
            if col == config.ID_COLUMN:
                continue

            unique_values = _unique_non_null(group[col])

            if len(unique_values) == 0:
                record[col] = np.nan
            elif len(unique_values) == 1:
                record[col] = unique_values[0]
            else:
                if col in config.STRICT_DUPLICATE_COLUMNS:
                    raise ValueError(
                        "True conflicting duplicate values found for "
                        f"student_id={student_id}, column='{col}': "
                        f"{unique_values}"
                    )

                conflict_records.append(
                    {
                        "student_id": student_id,
                        "column": col,
                        "values": repr(unique_values),
                    }
                )
                record[col] = unique_values[0]

        clean_records.append(record)

    clean_df = pd.DataFrame(clean_records)
    clean_df = clean_df[
        [c for c in original_columns if c in clean_df.columns]
    ].reset_index(drop=True)

    conflict_df = pd.DataFrame(
        conflict_records,
        columns=["student_id", "column", "values"],
    )

    return clean_df, conflict_df


def validate_cca(df: pd.DataFrame) -> None:
    """
    Ensure CCA='None' remains a valid category and no unexpected labels remain.
    """
    if "CCA" not in df.columns:
        raise ValueError("Required CCA column is missing.")

    if df["CCA"].isna().any():
        raise ValueError(
            "CCA contains missing values after cleaning. "
            "The string 'None' is a valid category for no CCA."
        )

    observed = set(df["CCA"].astype(str).unique())
    unexpected = observed - config.VALID_CCA_VALUES

    if unexpected:
        raise ValueError(
            f"Unexpected CCA values remain after cleaning: {sorted(unexpected)}"
        )


def clean_raw_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply reusable cleaning rules derived from Task 1 EDA.
    """
    df = standardise_categorical_values(df)
    df = clean_invalid_age(df)

    df_clean, conflict_df = consolidate_duplicate_students(df)

    validate_cca(df_clean)

    return df_clean, conflict_df
