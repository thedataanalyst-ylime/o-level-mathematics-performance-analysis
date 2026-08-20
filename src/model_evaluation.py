from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_validate

from . import config


def calculate_regression_metrics(y_true, y_pred) -> dict:
    """
    MAE: average absolute error in score points.
    RMSE: penalises larger errors more strongly.
    R2: proportion of target variance explained by the model.
    """
    return {
        "MAE": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "RMSE": float(
            mean_squared_error(y_true, y_pred) ** 0.5
        ),
        "R2": float(
            r2_score(y_true, y_pred)
        ),
    }


def build_shuffled_cv(
    random_state: int | None = None,
    folds: int | None = None,
) -> KFold:
    return KFold(
        n_splits=folds or config.CV_FOLDS,
        shuffle=True,
        random_state=(
            config.RANDOM_STATE
            if random_state is None
            else random_state
        ),
    )


def evaluate_train_test(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
) -> dict:
    train_metrics = calculate_regression_metrics(
        y_train,
        model.predict(X_train),
    )
    test_metrics = calculate_regression_metrics(
        y_test,
        model.predict(X_test),
    )

    return {
        "train": train_metrics,
        "test": test_metrics,
        "r2_gap": (
            train_metrics["R2"]
            - test_metrics["R2"]
        ),
    }


def cross_validate_regressor(
    model,
    X_train,
    y_train,
    cv,
) -> tuple[pd.DataFrame, dict]:
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    results = cross_validate(
        clone(model),
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=1,
    )

    n_folds = len(results["test_r2"])

    fold_table = pd.DataFrame(
        {
            "Fold": range(1, n_folds + 1),
            "Train_MAE": -results["train_mae"],
            "Validation_MAE": -results["test_mae"],
            "Train_RMSE": -results["train_rmse"],
            "Validation_RMSE": -results["test_rmse"],
            "Train_R2": results["train_r2"],
            "Validation_R2": results["test_r2"],
        }
    )

    summary = {
        "Mean_Validation_MAE": float(
            fold_table["Validation_MAE"].mean()
        ),
        "Mean_Validation_RMSE": float(
            fold_table["Validation_RMSE"].mean()
        ),
        "Mean_Training_R2": float(
            fold_table["Train_R2"].mean()
        ),
        "Mean_Validation_R2": float(
            fold_table["Validation_R2"].mean()
        ),
        "Std_Validation_R2": float(
            fold_table["Validation_R2"].std(ddof=0)
        ),
        "Mean_CV_R2_Gap": float(
            fold_table["Train_R2"].mean()
            - fold_table["Validation_R2"].mean()
        ),
    }

    return fold_table, summary


def residual_diagnostics(
    y_true,
    y_pred,
) -> tuple[pd.DataFrame, dict]:
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    residuals = y_true - y_pred
    absolute_errors = residuals.abs()

    detail = pd.DataFrame(
        {
            "Actual": y_true,
            "Predicted": y_pred,
            "Residual": residuals,
            "Absolute_Error": absolute_errors,
        }
    )

    summary = {
        "Mean_Residual": float(residuals.mean()),
        "Median_Residual": float(residuals.median()),
        "Residual_25pct": float(
            residuals.quantile(0.25)
        ),
        "Residual_75pct": float(
            residuals.quantile(0.75)
        ),
        "Within_5_Points_Pct": float(
            (absolute_errors <= 5).mean() * 100
        ),
        "Within_10_Points_Pct": float(
            (absolute_errors <= 10).mean() * 100
        ),
        "Error_Above_15_Points_Pct": float(
            (absolute_errors > 15).mean() * 100
        ),
        "Largest_Absolute_Error": float(
            absolute_errors.max()
        ),
    }

    return detail, summary


def get_aggregated_feature_importance(
    model,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]

    names = preprocessor.get_feature_names_out()

    transformed = pd.DataFrame(
        {
            "Transformed_Feature": names,
            "Importance": regressor.feature_importances_,
        }
    )
    transformed["Importance_%"] = (
        transformed["Importance"] * 100
    )

    def original_name(name: str) -> str:
        clean = name.split("__", 1)[-1]

        if clean in config.NUMERICAL_FEATURES:
            return clean

        for feature in config.CATEGORICAL_FEATURES:
            if clean == feature or clean.startswith(feature + "_"):
                return feature

        return clean

    transformed["Original_Feature"] = (
        transformed["Transformed_Feature"]
        .apply(original_name)
    )

    aggregated = (
        transformed
        .groupby("Original_Feature", as_index=False)["Importance"]
        .sum()
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    aggregated["Importance_%"] = aggregated["Importance"] * 100
    aggregated["Cumulative_Importance_%"] = (
        aggregated["Importance_%"].cumsum()
    )

    return transformed, aggregated


def academic_support_flags(
    predictions,
    threshold: float,
) -> pd.Series:
    """
    Illustrative screening flag only, not an automated support decision.
    """
    return pd.Series(predictions) < threshold
