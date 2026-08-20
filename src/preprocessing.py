from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import config


def build_preprocessor(
    numerical_imputation_strategy: str | None = None,
) -> ColumnTransformer:
    """
    Reusable modelling preprocessing.

    Numerical:
      median imputation by default (configurable)

    Categorical:
      one-hot encoding with unknown-category tolerance
    """
    strategy = (
        numerical_imputation_strategy
        or config.NUMERICAL_IMPUTATION_STRATEGY
    )

    numerical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy=strategy),
            ),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_transformer,
                config.NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_transformer,
                config.CATEGORICAL_FEATURES,
            ),
        ]
    )
