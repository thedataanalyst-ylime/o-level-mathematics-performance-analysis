from __future__ import annotations

import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from . import config
from .preprocessing import build_preprocessor


def get_candidate_models() -> dict:
    """
    At least three models, matching the notebook experimentation.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            random_state=config.RANDOM_STATE,
        ),
        "Random Forest": RandomForestRegressor(
            **config.BASELINE_RF_PARAMS,
        ),
    }


def build_model_pipeline(
    estimator,
    numerical_imputation_strategy: str | None = None,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numerical_imputation_strategy
                ),
            ),
            ("regressor", estimator),
        ]
    )


def run_baseline_experiments(
    X_train,
    X_test,
    y_train,
    y_test,
    metric_fn,
    numerical_imputation_strategy: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Fit Linear Regression, Decision Tree and Random Forest and compare
    train/test MAE, RMSE, R2 and generalisation gap.
    """
    rows = []
    fitted_models = {}

    for name, estimator in get_candidate_models().items():
        model = build_model_pipeline(
            estimator,
            numerical_imputation_strategy,
        )
        model.fit(X_train, y_train)

        train_metrics = metric_fn(
            y_train,
            model.predict(X_train),
        )
        test_metrics = metric_fn(
            y_test,
            model.predict(X_test),
        )

        rows.append(
            {
                "Model": name,
                "Train_MAE": train_metrics["MAE"],
                "Train_RMSE": train_metrics["RMSE"],
                "Train_R2": train_metrics["R2"],
                "Test_MAE": test_metrics["MAE"],
                "Test_RMSE": test_metrics["RMSE"],
                "Test_R2": test_metrics["R2"],
                "R2_Gap": (
                    train_metrics["R2"]
                    - test_metrics["R2"]
                ),
            }
        )

        fitted_models[name] = model

    comparison = (
        pd.DataFrame(rows)
        .sort_values("Test_R2", ascending=False)
        .reset_index(drop=True)
    )

    return comparison, fitted_models


def build_random_search_space() -> dict:
    """
    RandomizedSearchCV search space used in the notebook.
    """
    return {
        "regressor__n_estimators": randint(100, 501),
        "regressor__max_depth": config.MAX_DEPTH_OPTIONS,
        "regressor__min_samples_split": randint(2, 21),
        "regressor__min_samples_leaf": randint(1, 11),
        "regressor__max_features": config.MAX_FEATURES_OPTIONS,
    }


def run_random_forest_search(
    X_train,
    y_train,
    cv,
    numerical_imputation_strategy: str | None = None,
) -> RandomizedSearchCV:
    """
    Optimise Random Forest with 40 candidates x 5-fold CV and R2 scoring.
    """
    baseline_rf = RandomForestRegressor(
        **config.BASELINE_RF_PARAMS
    )

    pipeline = build_model_pipeline(
        baseline_rf,
        numerical_imputation_strategy,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=build_random_search_space(),
        n_iter=config.N_ITER_SEARCH,
        scoring="r2",
        cv=cv,
        random_state=config.RANDOM_STATE,
        n_jobs=config.SEARCH_N_JOBS,
        verbose=1,
        return_train_score=True,
        refit=True,
    )

    search.fit(X_train, y_train)
    return search


def build_known_tuned_random_forest(
    numerical_imputation_strategy: str | None = None,
) -> Pipeline:
    """
    Build the already validated tuned Random Forest from Task 1/notebook.
    """
    estimator = RandomForestRegressor(
        **config.FINAL_RF_PARAMS
    )

    return build_model_pipeline(
        estimator,
        numerical_imputation_strategy,
    )
