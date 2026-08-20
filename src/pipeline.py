from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .data_ingestion import load_source_data
from .data_preprocessing import clean_raw_data
from .feature_engineering import engineer_features
from .model_evaluation import (
    academic_support_flags,
    build_shuffled_cv,
    calculate_regression_metrics,
    cross_validate_regressor,
    evaluate_train_test,
    get_aggregated_feature_importance,
    residual_diagnostics,
)
from .model_experiment import (
    build_known_tuned_random_forest,
    run_baseline_experiments,
    run_random_forest_search,
)
from .reporting import save_dataframe, save_json
from .visualization import (
    save_actual_vs_predicted,
    save_aggregated_feature_importance,
    save_model_comparison,
    save_residual_plot,
)


def prepare_modelling_data(
    df_clean: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Create df_model, X and y using the Task 1-selected feature set.
    """
    df_model = (
        df_clean
        .dropna(subset=[config.TARGET])
        .copy()
    )

    missing_required = sorted(
        set(config.SELECTED_FEATURES)
        - set(df_model.columns)
    )

    if missing_required:
        raise ValueError(
            f"Missing model features: {missing_required}"
        )

    X = df_model[config.SELECTED_FEATURES].copy()
    y = pd.to_numeric(
        df_model[config.TARGET],
        errors="raise",
    ).copy()

    return df_model, X, y


def run_training_pipeline(
    database_path: str | Path,
    table_name: str | None,
    run_tuning: bool,
    random_state: int,
    test_size: float,
    numerical_imputation_strategy: str,
    support_threshold: float,
) -> dict:
    """
    Execute the complete Task 2 pipeline.
    """
    # 1. Ingestion
    raw_df, resolved_table = load_source_data(
        database_path,
        table_name,
    )

    # 2. Data cleaning
    cleaned_df, duplicate_conflicts = clean_raw_data(raw_df)

    # 3. Feature engineering
    cleaned_df = engineer_features(cleaned_df)

    # 4. Modelling data
    df_model, X, y = prepare_modelling_data(cleaned_df)

    # Record keeping
    if config.SAVE_INTERMEDIATE_CSV:
        save_dataframe(cleaned_df, config.CLEAN_DATA_PATH)
        save_dataframe(df_model, config.MODEL_DATA_PATH)

        if not duplicate_conflicts.empty:
            save_dataframe(
                duplicate_conflicts,
                config.OUTPUT_DIR / "duplicate_conflicts.csv",
            )

    data_summary = {
        "database": str(database_path),
        "table": resolved_table,
        "raw_rows": int(len(raw_df)),
        "clean_rows": int(len(cleaned_df)),
        "unique_students": int(
            cleaned_df[config.ID_COLUMN].nunique()
        ),
        "model_rows": int(len(df_model)),
        "rows_removed_missing_target": int(
            len(cleaned_df) - len(df_model)
        ),
        "missing_attendance_rate_model": int(
            df_model["attendance_rate"].isna().sum()
        ),
        "cca_counts_model": {
            str(k): int(v)
            for k, v in (
                df_model["CCA"]
                .value_counts(dropna=False)
                .to_dict()
                .items()
            )
        },
    }

    # 5. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # 6-7. Preprocessing + baseline experimentation
    comparison, fitted_models = run_baseline_experiments(
        X_train,
        X_test,
        y_train,
        y_test,
        metric_fn=calculate_regression_metrics,
        numerical_imputation_strategy=numerical_imputation_strategy,
    )

    # Baseline RF CV
    baseline_rf = fitted_models["Random Forest"]
    baseline_cv = build_shuffled_cv(
        random_state=random_state,
        folds=config.CV_FOLDS,
    )

    baseline_cv_folds, baseline_cv_summary = cross_validate_regressor(
        baseline_rf,
        X_train,
        y_train,
        cv=baseline_cv,
    )

    # 8. Optimisation
    tuning_cv = build_shuffled_cv(
        random_state=random_state,
        folds=config.CV_FOLDS,
    )

    if run_tuning:
        search = run_random_forest_search(
            X_train,
            y_train,
            cv=tuning_cv,
            numerical_imputation_strategy=numerical_imputation_strategy,
        )
        final_model = search.best_estimator_
        tuning_summary = {
            "mode": "RandomizedSearchCV",
            "n_candidates": config.N_ITER_SEARCH,
            "cv_folds": config.CV_FOLDS,
            "best_mean_cv_r2": float(search.best_score_),
            "best_params": search.best_params_,
        }
    else:
        final_model = build_known_tuned_random_forest(
            numerical_imputation_strategy
        )
        final_model.fit(X_train, y_train)

        tuning_summary = {
            "mode": "Known tuned parameters from Task 1/notebook",
            "best_params": config.FINAL_RF_PARAMS,
        }

    # 9. Final evaluation
    final_metrics = evaluate_train_test(
        final_model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Final CV stability check
    final_cv_folds, final_cv_summary = cross_validate_regressor(
        final_model,
        X_train,
        y_train,
        cv=config.CV_FOLDS,
    )

    y_test_pred = final_model.predict(X_test)

    residual_detail, residual_summary = residual_diagnostics(
        y_test,
        y_test_pred,
    )

    transformed_importance, aggregated_importance = (
        get_aggregated_feature_importance(final_model)
    )

    support_flags = academic_support_flags(
        y_test_pred,
        support_threshold,
    )

    prediction_output = residual_detail.copy()
    prediction_output["Academic_Support_Flag"] = support_flags.to_numpy()

    # Save fitted model
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, config.MODEL_PATH)

    # Save tabular outputs
    save_dataframe(
        comparison,
        config.OUTPUT_DIR / "model_comparison.csv",
    )
    save_dataframe(
        baseline_cv_folds,
        config.OUTPUT_DIR / "baseline_rf_cv_folds.csv",
    )
    save_dataframe(
        final_cv_folds,
        config.OUTPUT_DIR / "tuned_rf_cv_folds.csv",
    )
    save_dataframe(
        prediction_output,
        config.OUTPUT_DIR / "test_predictions_and_residuals.csv",
    )
    save_dataframe(
        transformed_importance,
        config.OUTPUT_DIR / "transformed_feature_importance.csv",
    )
    save_dataframe(
        aggregated_importance,
        config.OUTPUT_DIR / "aggregated_feature_importance.csv",
    )

    # Save JSON summaries
    save_json(
        data_summary,
        config.OUTPUT_DIR / "data_summary.json",
    )
    save_json(
        tuning_summary,
        config.OUTPUT_DIR / "tuning_summary.json",
    )
    save_json(
        final_metrics,
        config.OUTPUT_DIR / "final_model_metrics.json",
    )
    save_json(
        baseline_cv_summary,
        config.OUTPUT_DIR / "baseline_rf_cv_summary.json",
    )
    save_json(
        final_cv_summary,
        config.OUTPUT_DIR / "tuned_rf_cv_summary.json",
    )
    save_json(
        residual_summary,
        config.OUTPUT_DIR / "residual_diagnostics_summary.json",
    )

    chart_paths = []

    if config.SAVE_CHARTS:
        chart_paths = [
            save_model_comparison(comparison),
            save_actual_vs_predicted(y_test, y_test_pred),
            save_residual_plot(
                y_test_pred,
                residual_detail["Residual"],
            ),
            save_aggregated_feature_importance(
                aggregated_importance
            ),
        ]

    return {
        "data_summary": data_summary,
        "comparison": comparison,
        "baseline_cv_summary": baseline_cv_summary,
        "tuning_summary": tuning_summary,
        "final_metrics": final_metrics,
        "final_cv_summary": final_cv_summary,
        "residual_summary": residual_summary,
        "aggregated_importance": aggregated_importance,
        "model_path": config.MODEL_PATH,
        "chart_paths": chart_paths,
    }
