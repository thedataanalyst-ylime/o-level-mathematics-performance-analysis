from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src import config
from src.data_preprocessing import standardise_categorical_values
from src.feature_engineering import engineer_features
from src.model_evaluation import academic_support_flags


def prepare_prediction_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Allow prediction from either:
    - already engineered features, or
    - raw columns containing sleep_time/wake_time and n_male/n_female.
    """
    df = standardise_categorical_values(df)

    if "sleep_duration" not in df.columns or "class_size" not in df.columns:
        df = engineer_features(df)

    missing = sorted(
        set(config.SELECTED_FEATURES)
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Prediction input is missing required fields: {missing}"
        )

    return df[config.SELECTED_FEATURES].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate O-Level Mathematics predictions from a saved model."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV containing student predictor data.",
    )

    parser.add_argument(
        "--output",
        default="outputs/new_student_predictions.csv",
        help="Output CSV path.",
    )

    parser.add_argument(
        "--support-threshold",
        type=float,
        default=config.SUPPORT_THRESHOLD,
        help="Illustrative academic-support screening threshold.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)
    X_new = prepare_prediction_data(df)

    model = joblib.load(config.MODEL_PATH)
    predictions = model.predict(X_new)

    result = df.copy()
    result["predicted_final_test"] = predictions
    result["academic_support_flag"] = (
        academic_support_flags(
            predictions,
            args.support_threshold,
        )
        .to_numpy()
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(
        result[
            [
                "predicted_final_test",
                "academic_support_flag",
            ]
        ].head()
    )
    print(f"\nPredictions saved to: {output_path}")


if __name__ == "__main__":
    main()
