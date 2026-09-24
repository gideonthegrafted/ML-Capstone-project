"""Project-wide configuration: paths, seed, run mode and dataset contracts.

Every other module imports its paths from here, so nothing depends on the
current working directory. All paths are derived from this file's location.

The dataset contracts below were established by inspecting the actual CSV
files in Phase 1 (shape, columns, checksums). If a file on disk does not match
its contract, loading fails loudly instead of silently continuing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths (all relative to the project root; never hard-code absolute paths)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                # immutable source CSVs (git-ignored)
PROCESSED_DIR = DATA_DIR / "processed"    # regenerable derived artefacts
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
TUNING_DIR = RESULTS_DIR / "tuning"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# --------------------------------------------------------------------------
# Reproducibility and evaluation protocol
# --------------------------------------------------------------------------
RANDOM_STATE = 42     # course guideline 7.1: random_state=42 wherever applicable
TEST_SIZE = 0.20      # course guideline 7.2: consistent 80:20 split per track
CV_FOLDS = 5          # course guideline 3.1: 5-fold cross-validation

# Regression stratification (see docs/clarifications.md, item C1).
# The rubric asks for a stratified split, but the regression target is
# continuous. The target is binned into quantiles ONLY to build the split;
# the bins are never a model feature and the target itself is never changed.
# Set to None to disable and fall back to a plain random split.
REGRESSION_STRATIFY_BINS: int | None = 10

# Run mode. "full" produces the reported results. "dev" is for quick passes:
# every artefact gets the "_dev" suffix so it can never be mistaken for a
# reported figure. Set with the environment variable ML_CAPSTONE_MODE.
RUN_MODE = os.environ.get("ML_CAPSTONE_MODE", "full").strip().lower()
if RUN_MODE not in {"full", "dev"}:
    raise ValueError(f"ML_CAPSTONE_MODE must be 'full' or 'dev', got {RUN_MODE!r}")
OUTPUT_SUFFIX = "" if RUN_MODE == "full" else "_dev"

# Saved models larger than this are flagged by the validation script.
MAX_MODEL_FILE_MB = 50

# --------------------------------------------------------------------------
# Classification label definition (approved decision, docs/clarifications.md C3/C4)
# --------------------------------------------------------------------------
RATING_POSITIVE_MIN = 4.0   # rating >= 4  -> "positive"
RATING_NEGATIVE_MAX = 2.0   # rating <= 2  -> "negative"
# Ratings strictly between 2 and 4 (2.5, 3, 3.5) are ambiguous and dropped.
LABEL_POSITIVE = "positive"
LABEL_NEGATIVE = "negative"
CLASS_LABELS = (LABEL_NEGATIVE, LABEL_POSITIVE)   # fixed display order
# Binary precision/recall are computed for the minority class, "negative".
POS_LABEL = LABEL_NEGATIVE

# TF-IDF vocabulary cap. Deliberately NOT fixed yet: Phase 6 measures the real
# vocabulary size, memory and runtime before choosing it (approved decision 5).
TFIDF_MAX_FEATURES: int | None = None

# Upper bound for any sparse-to-dense conversion (GaussianNB needs dense input).
MAX_DENSE_MB = 400


# --------------------------------------------------------------------------
# Dataset contracts
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetContract:
    """What a raw dataset file must look like, and how it is used."""

    key: str
    filename: str
    task: str                       # "regression" | "classification" | "clustering (Review 2)"
    review: int                     # review in which the dataset is modelled
    sha256: str
    n_rows: int
    columns: tuple[str, ...]        # exact raw column order
    target: str | None = None
    numeric_features: tuple[str, ...] = ()
    categorical_features: tuple[str, ...] = ()
    text_features: tuple[str, ...] = ()
    excluded: dict[str, str] = field(default_factory=dict)   # column -> reason
    source_url: str = ""
    synthetic: bool | None = None

    @property
    def n_cols(self) -> int:
        return len(self.columns)

    @property
    def path(self) -> Path:
        return RAW_DIR / self.filename

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.numeric_features + self.categorical_features + self.text_features


STUDENT = DatasetContract(
    key="student_performance",
    filename="Student_Performance.csv",
    task="regression",
    review=1,
    sha256="93793b00d9026d0b4907df0ca9f88b3696747c7496679d35833bb0fbf9fb57cf",
    n_rows=10_000,
    columns=(
        "Hours Studied",
        "Previous Scores",
        "Extracurricular Activities",
        "Sleep Hours",
        "Sample Question Papers Practiced",
        "Performance Index",
    ),
    target="Performance Index",
    numeric_features=(
        "Hours Studied",
        "Previous Scores",
        "Sleep Hours",
        "Sample Question Papers Practiced",
    ),
    categorical_features=("Extracurricular Activities",),
    excluded={},   # no identifiers or leakage columns were found in Phase 1
    source_url="https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression",
    synthetic=True,   # stated by the dataset author on the Kaggle page
)

REVIEWS = DatasetContract(
    key="restaurant_reviews",
    filename="Restaurant reviews.csv",
    task="classification",
    review=1,
    sha256="43a3950890537219def4af1f7a4473dc0e6fa36dafb2df7a07da723c7fb71517",
    n_rows=10_000,
    columns=(
        "Restaurant",
        "Reviewer",
        "Review",
        "Rating",
        "Metadata",
        "Time",
        "Pictures",
        "7514",
    ),
    # The model target is derived from "Rating" (see RATING_* above); the raw
    # "Rating" column itself is the label source and never a model input.
    target="Rating",
    text_features=("Review",),
    excluded={
        "Rating": "Label source: the sentiment target is derived from it (direct leakage).",
        "Restaurant": "Identifier of the reviewed business; would let models learn restaurant-level rating bias instead of text sentiment.",
        "Reviewer": "Personal name (identifier and privacy concern); never shown or modelled.",
        "Metadata": "Reviewer activity counts, not review content. Available for EDA / team feature engineering only.",
        "Time": "Timestamp, not review content. Available for EDA only.",
        "Pictures": "Picture count, not review content. Available for EDA / team feature engineering only.",
        "7514": "Scraping artefact: 1 non-null value in 10,000 rows.",
    },
    source_url="https://www.kaggle.com/datasets/joebeachcapital/restaurant-reviews",
    synthetic=False,
)

CAFE = DatasetContract(
    key="cafe_sales",
    filename="Cafe_sales_cleaned.csv",
    task="clustering (Review 2)",
    review=2,
    sha256="41e0f4a082be47049d9ebc3a4e5a05e74e9458c1d987c4502c8a26b8c5bb66a6",
    n_rows=10_000,
    columns=(
        "Transaction ID",
        "Item",
        "Quantity",
        "Price Per Unit",
        "Total Spent",
        "Payment Method",
        "Location",
        "Transaction Date",
    ),
    excluded={"Transaction ID": "Unique identifier."},
    source_url="https://www.kaggle.com/datasets/ahmedmohamed2003/cafe-sales-dirty-data-for-cleaning-training",
    synthetic=True,   # stated by the dataset author; the supplied file is a derived cleaned version
)

CONTRACTS: dict[str, DatasetContract] = {c.key: c for c in (STUDENT, REVIEWS, CAFE)}

# Marker that identifies unwritten student-authored sections in the notebooks.
TEAM_MARKER = "TEAM ANALYSIS REQUIRED"
