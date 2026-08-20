from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import config


def _charts_dir(path: str | Path | None = None) -> Path:
    charts_dir = Path(path or config.CHARTS_DIR)
    charts_dir.mkdir(parents=True, exist_ok=True)
    return charts_dir


def save_model_comparison(
    comparison: pd.DataFrame,
    charts_dir: str | Path | None = None,
) -> Path:
    output = _charts_dir(charts_dir) / "ModelComparison_TestR2.png"

    plot_data = comparison.sort_values("Test_R2", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(
        plot_data["Model"],
        plot_data["Test_R2"],
    )
    plt.xlabel("Test R²")
    plt.ylabel("Model")
    plt.title("Model Comparison — Test R²")
    plt.tight_layout()
    plt.savefig(output, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close()

    return output


def save_actual_vs_predicted(
    y_true,
    y_pred,
    charts_dir: str | Path | None = None,
) -> Path:
    output = (
        _charts_dir(charts_dir)
        / "Actual_vs_Predicted_O-LevelMathScores.png"
    )

    min_score = min(min(y_true), min(y_pred))
    max_score = max(max(y_true), max(y_pred))

    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot(
        [min_score, max_score],
        [min_score, max_score],
        linestyle="--",
    )
    plt.xlabel("Actual Final Test Score")
    plt.ylabel("Predicted Final Test Score")
    plt.title("Actual vs Predicted O-Level Mathematics Scores")
    plt.tight_layout()
    plt.savefig(output, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close()

    return output


def save_residual_plot(
    predictions,
    residuals,
    charts_dir: str | Path | None = None,
) -> Path:
    output = (
        _charts_dir(charts_dir)
        / "ResidualPlot_TunedRandomForest.png"
    )

    plt.figure(figsize=(7, 6))
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, linestyle="--")
    plt.xlabel("Predicted Final Test Score")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual Plot — Tuned Random Forest")
    plt.tight_layout()
    plt.savefig(output, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close()

    return output


def save_aggregated_feature_importance(
    aggregated: pd.DataFrame,
    charts_dir: str | Path | None = None,
) -> Path:
    output = (
        _charts_dir(charts_dir)
        / "AggregatedFeatureImportance_TunedRandomForest.png"
    )

    plot_data = aggregated.sort_values(
        "Importance_%",
        ascending=True,
    )

    plt.figure(figsize=(10, 6))
    plt.barh(
        plot_data["Original_Feature"],
        plot_data["Importance_%"],
    )
    plt.xlabel("Aggregated Feature Importance (%)")
    plt.ylabel("Predictor")
    plt.title("Aggregated Feature Importance — Tuned Random Forest")
    plt.tight_layout()
    plt.savefig(output, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close()

    return output
