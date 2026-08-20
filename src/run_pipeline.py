from __future__ import annotations

import argparse
from pathlib import Path

from . import config
from .pipeline import run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the end-to-end O-Level Mathematics ML pipeline "
            "from SQLite ingestion through model evaluation."
        )
    )

    parser.add_argument(
        "--db",
        default=str(config.DATABASE_PATH),
        help="Path to the provided SQLite database.",
    )

    parser.add_argument(
        "--table",
        default=config.TABLE_NAME,
        help="SQLite table name. Auto-detected when only one table exists.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=config.RANDOM_STATE,
        help="Random seed used for train/test split and CV.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=config.TEST_SIZE,
        help="Test-set proportion. Default: 0.20.",
    )

    parser.add_argument(
        "--numeric-imputer",
        choices=["median", "mean"],
        default=config.NUMERICAL_IMPUTATION_STRATEGY,
        help="Numerical missing-value imputation strategy.",
    )

    parser.add_argument(
        "--support-threshold",
        type=float,
        default=config.SUPPORT_THRESHOLD,
        help="Illustrative predicted-score threshold for academic-support flag.",
    )

    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help=(
            "Skip RandomizedSearchCV and use the already validated "
            "Tuned Random Forest parameters from Task 1."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_tuning = (
        config.RUN_HYPERPARAMETER_SEARCH
        and not args.skip_tuning
    )

    print("\nDAB Challenge — End-to-End ML Pipeline")
    print("=" * 45)
    print(f"Database: {args.db}")
    print(f"Table: {args.table or 'auto-detect'}")
    print(f"Random state: {args.random_state}")
    print(f"Test size: {args.test_size}")
    print(f"Numerical imputer: {args.numeric_imputer}")
    print(f"Run tuning: {run_tuning}")
    print(f"Support threshold: {args.support_threshold}")

    results = run_training_pipeline(
        database_path=Path(args.db),
        table_name=args.table,
        run_tuning=run_tuning,
        random_state=args.random_state,
        test_size=args.test_size,
        numerical_imputation_strategy=args.numeric_imputer,
        support_threshold=args.support_threshold,
    )

    print("\n1. DATA SUMMARY")
    print(results["data_summary"])

    print("\n2. MODEL COMPARISON")
    print(
        results["comparison"]
        .round(3)
        .to_string(index=False)
    )

    print("\n3. BASELINE RANDOM FOREST CV")
    print(results["baseline_cv_summary"])

    print("\n4. RANDOM FOREST OPTIMISATION")
    print(results["tuning_summary"])

    print("\n5. FINAL MODEL METRICS")
    print(results["final_metrics"])

    print("\n6. FINAL MODEL CV")
    print(results["final_cv_summary"])

    print("\n7. RESIDUAL DIAGNOSTICS")
    print(results["residual_summary"])

    print("\n8. AGGREGATED FEATURE IMPORTANCE")
    print(
        results["aggregated_importance"]
        .round(3)
        .to_string(index=False)
    )

    print(f"\nSaved model: {results['model_path']}")

    if results["chart_paths"]:
        print("\nSaved PNG charts:")
        for path in results["chart_paths"]:
            print(f"  - {path}")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
