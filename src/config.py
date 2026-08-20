from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = PROJECT_ROOT / "data" / "score.db"
TABLE_NAME = None

CLEAN_DATA_PATH = PROJECT_ROOT / "outputs" / "df_clean.csv"
MODEL_DATA_PATH = PROJECT_ROOT / "outputs" / "df_model.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "tuned_random_forest_pipeline.joblib"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = PROJECT_ROOT / "charts"

RANDOM_STATE = 27
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET = "final_test"
ID_COLUMN = "student_id"

NUMERICAL_FEATURES = [
    "class_size",
    "number_of_siblings",
    "attendance_rate",
    "sleep_duration",
    "hours_per_week",
]

CATEGORICAL_FEATURES = [
    "direct_admission",
    "CCA",
    "learning_style",
    "tuition",
]

SELECTED_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

VALID_AGES = [15, 16]
VALID_CCA_VALUES = {"Arts", "Clubs", "None", "Sports"}
NUMERICAL_IMPUTATION_STRATEGY = "median"

STRICT_DUPLICATE_COLUMNS = {
    TARGET,
    "attendance_rate",
}

BASELINE_RF_PARAMS = {
    "n_estimators": 100,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

RUN_HYPERPARAMETER_SEARCH = True
N_ITER_SEARCH = 40
SEARCH_N_JOBS = 1

MAX_DEPTH_OPTIONS = [None, 5, 8, 10, 12, 15, 20]
MAX_FEATURES_OPTIONS = [1.0, "sqrt", "log2", 0.5, 0.7]

FINAL_RF_PARAMS = {
    "n_estimators": 384,
    "max_depth": 10,
    "max_features": 1.0,
    "min_samples_split": 11,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

SUPPORT_THRESHOLD = 60
SAVE_INTERMEDIATE_CSV = True
SAVE_CHARTS = True
CHART_DPI = 300
