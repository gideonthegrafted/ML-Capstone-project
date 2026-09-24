"""Verify what EXISTS in the project, not what was intended.

Each check reports one of:
    PASS     verified on disk / by recomputation
    FAIL     something is wrong - exit code 1
    PENDING  cannot be verified yet (e.g. a later build phase, or team work)

The final line states whether the project is submission-ready. It is never
ready while any check is PENDING - in particular while TEAM ANALYSIS REQUIRED
placeholders remain or no engineered feature has been registered.

    python scripts/validate_project.py            # exit 1 only on FAIL
    python scripts/validate_project.py --strict   # exit 1 on FAIL or PENDING
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

# --------------------------------------------------------------------------
# Scope guard: Review 2 work must not appear in Review 1 code.
# (Regression RandomForest/GradientBoosting REGRESSORS are in scope and are
#  deliberately not matched.)
# --------------------------------------------------------------------------
FORBIDDEN_PATTERNS = {
    "Part B: Random Forest Classifier": r"\bRandomForestClassifier\b",
    "Part B: AdaBoost": r"\bAdaBoost\w*",
    "Part B: Gradient Boosting Classifier": r"\b(Hist)?GradientBoostingClassifier\b",
    "Part B: Bagging": r"\bBagging\w*",
    "Part B: MLP": r"\bMLP\w*",
    "Part B: XGBoost/LightGBM classifier": r"\b(XGB|LGBM)Classifier\b",
    "Clustering: K-Means": r"\b(MiniBatch)?KMeans\b",
    "Clustering: Agglomerative": r"\bAgglomerativeClustering\b",
    "Clustering: sklearn.cluster": r"\bsklearn\.cluster\b",
    "Clustering: dendrogram/linkage": r"\b(dendrogram|scipy\.cluster)\b",
    "Clustering: t-SNE": r"\bTSNE\b",
}
# Files allowed to NAME the forbidden estimators (they define/test the guard).
SCOPE_GUARD_EXEMPT = {"scripts/validate_project.py", "tests/test_scope_guard.py"}

REQUIRED_FILES = [
    "README.md", "requirements.txt", ".gitignore",
    "notebooks/regression.ipynb", "notebooks/classification.ipynb",
    "src/__init__.py", "src/config.py", "src/data_loading.py", "src/preprocessing.py",
    "src/feature_engineering.py", "src/evaluation.py", "src/plotting.py",
    "src/regression_models.py", "src/classification_models.py",
    "scripts/run_all.py", "scripts/validate_project.py", "scripts/setup_data.py",
    "docs/rubric_checklist.md", "docs/dataset_sources.md", "docs/clarifications.md",
]
REQUIRED_DIRS = ["data/raw", "data/processed", "results/tables", "results/figures",
                 "results/tuning", "models", "tests"]


@dataclass
class Check:
    group: str
    name: str
    status: str   # PASS | FAIL | PENDING
    detail: str = ""


CHECKS: list[Check] = []


def add(group, name, status, detail=""):
    CHECKS.append(Check(group, name, status, detail))


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def code_sources():
    """Yield (label, code text) for every Python file and notebook code cell."""
    import nbformat
    for folder in ("src", "scripts", "tests", "app"):
        for py in sorted((ROOT / folder).rglob("*.py")) if (ROOT / folder).exists() else []:
            yield rel(py), py.read_text(encoding="utf-8")
    for nb_path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = nbformat.read(nb_path, as_version=4)
        code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
        yield rel(nb_path), code


# --------------------------------------------------------------------------
# Check groups
# --------------------------------------------------------------------------
def check_structure():
    missing = [f for f in REQUIRED_FILES if not (ROOT / f).is_file()]
    missing += [d + "/" for d in REQUIRED_DIRS if not (ROOT / d).is_dir()]
    add("Structure", "required files and folders",
        "FAIL" if missing else "PASS", "missing: " + ", ".join(missing) if missing else f"{len(REQUIRED_FILES)} files, {len(REQUIRED_DIRS)} folders")


def check_environment():
    ok_py = sys.version_info >= (3, 11)
    add("Environment", "Python >= 3.11", "PASS" if ok_py else "FAIL", sys.version.split()[0])
    mismatches = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "==" not in line:
            continue
        name, pinned = line.split("==")
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = "not installed"
        if installed != pinned:
            mismatches.append(f"{name} pinned {pinned}, installed {installed}")
    add("Environment", "installed versions match requirements.txt",
        "FAIL" if mismatches else "PASS", "; ".join(mismatches) or "all pins satisfied")


def check_raw_data():
    from src.data_loading import ContractError, load_raw
    for contract in config.CONTRACTS.values():
        try:
            df = load_raw(contract)
            add("Raw data", f"{contract.filename}", "PASS",
                f"sha256 OK, {df.shape[0]} x {df.shape[1]} matches contract")
        except FileNotFoundError:
            add("Raw data", contract.filename, "FAIL", "missing - run scripts/setup_data.py")
        except ContractError as exc:
            add("Raw data", contract.filename, "FAIL", str(exc)[:300])


def check_exclusions():
    for contract in (config.STUDENT, config.REVIEWS):
        leaked = set(contract.feature_columns) & set(contract.excluded)
        target_in_x = contract.target in contract.feature_columns
        bad = sorted(leaked) + ([contract.target] if target_in_x else [])
        add("Exclusions", f"{contract.key}: no excluded/target column among features",
            "FAIL" if bad else "PASS", ", ".join(bad) or f"features = {list(contract.feature_columns)}")


def check_catalogue():
    from src.classification_models import CLASSIFICATION_PART_A
    from src.regression_models import REGRESSION_ALGORITHMS
    n_reg, n_clf = len(REGRESSION_ALGORITHMS), len(CLASSIFICATION_PART_A)
    add("Algorithms", "regression catalogue has 10 algorithms", "PASS" if n_reg == 10 else "FAIL", str(n_reg))
    add("Algorithms", "classification catalogue has exactly 5 Part A algorithms",
        "PASS" if n_clf == 5 else "FAIL", str(n_clf))


def check_scope_guard():
    hits = []
    for label, text in code_sources():
        if label in SCOPE_GUARD_EXEMPT:
            continue
        for desc, pattern in FORBIDDEN_PATTERNS.items():
            if re.search(pattern, text):
                hits.append(f"{label}: {desc}")
    if (ROOT / "notebooks" / "clustering.ipynb").exists():
        hits.append("notebooks/clustering.ipynb exists (Review 2)")
    import src.classification_models as cm
    extra = [n for n in dir(cm) if re.search(r"part_?b", n, re.I)]
    hits += [f"src/classification_models.py defines {n}" for n in extra]
    add("Scope guard", "no Part B / clustering code", "FAIL" if hits else "PASS",
        "; ".join(hits) or "no forbidden estimators, modules or builders found")


def check_notebooks():
    import nbformat
    for name in ("regression.ipynb", "classification.ipynb"):
        path = ROOT / "notebooks" / name
        if not path.exists():
            add("Notebooks", name, "FAIL", "missing")
            continue
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        code = [c for c in nb.cells if c.cell_type == "code"]
        unexecuted = [i for i, c in enumerate(code) if c.execution_count is None]
        errors = [o.get("ename") for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
        counts = [c.execution_count for c in code if c.execution_count is not None]
        in_order = counts == sorted(counts) and len(set(counts)) == len(counts)
        if errors:
            add("Notebooks", f"{name} executed without errors", "FAIL", f"error outputs: {errors}")
        elif unexecuted or not code:
            add("Notebooks", f"{name} executed without errors", "PENDING",
                f"{len(unexecuted)} of {len(code)} code cells not executed")
        elif not in_order:
            add("Notebooks", f"{name} executed top to bottom", "FAIL", "execution counts out of order")
        else:
            add("Notebooks", f"{name} executed top to bottom without errors", "PASS",
                f"{len(code)} code cells, {len(nb.cells) - len(code)} markdown cells")


def check_models():
    files = sorted(config.MODELS_DIR.glob("*.joblib"))
    if not files:
        add("Saved models", "pipelines saved in models/", "PENDING", "no saved models yet (Phases 5 and 8)")
        return
    import joblib
    from sklearn.pipeline import Pipeline
    big = [f"{f.name} ({f.stat().st_size / 1e6:.1f} MB)" for f in files
           if f.stat().st_size / 1e6 > config.MAX_MODEL_FILE_MB]
    add("Saved models", f"file sizes <= {config.MAX_MODEL_FILE_MB} MB", "FAIL" if big else "PASS",
        ", ".join(big) or f"{len(files)} files")
    not_pipe = [f.name for f in files if not isinstance(joblib.load(f), Pipeline)]
    add("Saved models", "every saved model is a Pipeline carrying its preprocessing",
        "FAIL" if not_pipe else "PASS", ", ".join(not_pipe) or f"{len(files)} pipelines")


REGRESSION_TABLES = [
    "regression_descriptive_statistics", "regression_outlier_check", "regression_leakage_identifier_check",
    "regression_split_target_summary", "regression_linear_coefficients", "regression_polynomial_degrees",
    "regression_comparison", "regression_comparison_details", "regression_cv_all_models",
    "regression_tuning_summary", "regression_tuning_test_before_after", "regression_tuned_results",
    "regression_final_comparison",
]
# Fixed names, plus glob patterns for the diagnostics (their names depend on the nominated models).
REGRESSION_FIGURES = [
    "regression_target_distribution.png", "regression_feature_distributions.png",
    "regression_outlier_boxplots.png", "regression_correlation_heatmap.png",
    "regression_scatter_feature_target.png", "regression_boxplots_by_feature.png",
    "regression_linear_coefficients.png", "regression_random_forest_feature_importance.png",
    "regression_decision_tree_feature_importance.png", "regression_comparison_r2.png",
    "regression_cv_vs_test_r2.png", "regression_diag_baseline_vs_tuned.png",
    "regression_diag_pred_vs_actual_*.png", "regression_diag_residuals_*.png",
]
DATA_FUNCTIONS = ("cross_validate", "cross_val_score", "cross_val_predict", "validation_curve",
                  "learning_curve", "GridSearchCV", "RandomizedSearchCV")


def calls_using_test_data(code: str) -> list[str]:
    """Model-selection calls (CV, curves, searches) whose arguments mention the test set,
    and any .fit(...) on test data. Returns the offending call texts."""
    hits = []
    for fn in DATA_FUNCTIONS:
        for m in re.finditer(rf"\b{fn}\(", code):
            depth, j = 1, m.end()
            while j < len(code) and depth:
                depth += {"(": 1, ")": -1}.get(code[j], 0)
                j += 1
            call = code[m.start():j]
            if re.search(r"\b[Xy]_test\b", call):
                hits.append(" ".join(call.split())[:120])
    hits += [m.group(0) for m in re.finditer(r"\.fit\(\s*[Xy]_test\b", code)]
    return hits


def rebuild_regression_split():
    """Rebuild the regression split from the raw file, independently of the notebook."""
    from src.data_loading import load_raw
    from src.preprocessing import drop_exact_duplicates, split_regression
    c = config.STUDENT
    df, _ = drop_exact_duplicates(load_raw(c))
    X = df[list(c.numeric_features + c.categorical_features)]
    return split_regression(X, df[c.target])


def check_regression_artifacts():
    import hashlib

    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline

    from src.evaluation import regression_metrics
    from src.regression_models import REGRESSION_ALGORITHMS

    G = "Regression results"
    missing_t = [t for t in REGRESSION_TABLES
                 if not ((config.TABLES_DIR / f"{t}.csv").exists() and (config.TABLES_DIR / f"{t}.md").exists())]
    add(G, f"required tables exist ({len(REGRESSION_TABLES)}, CSV + Markdown)", "FAIL" if missing_t else "PASS",
        ", ".join(missing_t) or "all present")
    missing_f = [f for f in REGRESSION_FIGURES if not list(config.FIGURES_DIR.glob(f))]
    add(G, f"required figures exist ({len(REGRESSION_FIGURES)} names/patterns)", "FAIL" if missing_f else "PASS",
        ", ".join(missing_f) or f"{len(list(config.FIGURES_DIR.glob('regression_*.png')))} regression figures on disk")

    nb_path = ROOT / "notebooks" / "regression.ipynb"
    import nbformat
    code = "\n".join(c.source for c in nbformat.read(nb_path, as_version=4).cells if c.cell_type == "code")
    bad = calls_using_test_data(code)
    add(G, "CV, hyperparameter curves and tuning use training data only (static scan)",
        "FAIL" if bad else "PASS", "; ".join(bad) or "no CV/search/curve call and no .fit() receives X_test / y_test")

    manifest_path = config.MODELS_DIR / "regression_manifest.json"
    if not manifest_path.exists():
        add(G, "saved regression pipelines", "PENDING", "models/regression_manifest.json not produced yet")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    X_train, X_test, y_train, y_test = rebuild_regression_split()
    fp = hashlib.sha256(",".join(map(str, sorted(X_test.index))).encode()).hexdigest()
    add(G, "all saved models share the common split (independently rebuilt)",
        "PASS" if fp == manifest["test_index_sha256"] else "FAIL",
        f"test rows {len(X_test)}, fingerprint {'matches' if fp == manifest['test_index_sha256'] else 'DIFFERS'}")

    baseline = [m for m in manifest["models"] if m["variant"] == "baseline"]
    names = sorted(m["model"] for m in baseline)
    add(G, "exactly the 10 required regressors are saved (baseline)",
        "PASS" if names == sorted(a.name for a in REGRESSION_ALGORITHMS) else "FAIL", f"{len(baseline)} baseline models")

    reload_diff, refit_diff, not_pipe, leaks, wrong_cls = 0.0, 0.0, [], [], []
    catalogue = {a.name: a.estimator for a in REGRESSION_ALGORITHMS}
    for entry in manifest["models"]:
        pipe = joblib.load(config.MODELS_DIR / entry["file"])
        if not isinstance(pipe, Pipeline):
            not_pipe.append(entry["file"])
            continue
        est = pipe.named_steps["model"]
        cls = "PolynomialFeatures+LinearRegression" if isinstance(est, Pipeline) else type(est).__name__
        if cls != catalogue[entry["model"]]:
            wrong_cls.append(f"{entry['file']}: {cls}")
        m = regression_metrics(y_test, pipe.predict(X_test))
        reload_diff = max(reload_diff, *(abs(m[k] - entry[k]) for k in ("R2", "RMSE", "MAE")))
        refit = clone(pipe).fit(X_train, y_train)
        m2 = regression_metrics(y_test, refit.predict(X_test))
        refit_diff = max(refit_diff, *(abs(m2[k] - entry[k]) for k in ("R2", "RMSE", "MAE")))
        scaler = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps.get("scale")
        n_num = len(config.STUDENT.numeric_features)
        if scaler is not None and not np.allclose(scaler.mean_[:n_num], X_train[list(config.STUDENT.numeric_features)].mean()):
            leaks.append(entry["file"])
    add(G, "every saved model is a Pipeline carrying its preprocessing", "FAIL" if not_pipe else "PASS",
        ", ".join(not_pipe) or f"{len(manifest['models'])} pipelines")
    add(G, "saved models use the catalogue's estimator classes", "FAIL" if wrong_cls else "PASS",
        ", ".join(wrong_cls) or "all match")
    add(G, "saved pipelines reload and reproduce their recorded metrics", "PASS" if reload_diff <= 1e-9 else "FAIL",
        f"largest difference {reload_diff:.1e}")
    add(G, "metrics reproduce on an independent refit from scratch", "PASS" if refit_diff <= 1e-9 else "FAIL",
        f"largest difference {refit_diff:.1e} over {len(manifest['models'])} refits")
    add(G, "no preprocessing leakage: every scaler holds TRAINING-set means", "FAIL" if leaks else "PASS",
        ", ".join(leaks) or "all scalers match the training-set means")

    table = pd.read_csv(config.TABLES_DIR / "regression_comparison.csv", index_col=0)
    tdiff = max(abs(table.loc[e["model"], col] - e[key]) for e in baseline
                for col, key in (("R²", "R2"), ("RMSE", "RMSE"), ("MAE", "MAE")))
    tuned = [m for m in manifest["models"] if m["variant"] == "tuned"]
    ttable = pd.read_csv(config.TABLES_DIR / "regression_tuned_results.csv", index_col=0)
    tdiff2 = max([abs(ttable.loc[e["model"], col] - e[key]) for e in tuned
                  for col, key in (("Tuned test R²", "R2"), ("Tuned test RMSE", "RMSE"), ("Tuned test MAE", "MAE"))] or [0.0])
    add(G, "comparison and tuned tables agree with the saved models", "PASS" if max(tdiff, tdiff2) <= 1e-9 else "FAIL",
        f"largest difference {max(tdiff, tdiff2):.1e}; {len(tuned)} tuned model(s)")
    add(G, "formal tuning applied to at least 2 models (rubric C3)", "PASS" if len(tuned) >= 2 else "FAIL",
        ", ".join(m["model"] for m in tuned))


def check_results():
    check_regression_artifacts()
    table = config.TABLES_DIR / "classification_comparison.csv"
    add("Classification results", "classification comparison table", "PASS" if table.exists() else "PENDING",
        rel(table) if table.exists() else "not produced yet (Phases 7-8)")
    add("Classification results", "classification metrics reproduce on independent refit", "PENDING",
        "implemented in Phase 8")


def check_team_sections():
    import nbformat
    from src import feature_engineering
    total = 0
    for name in ("regression.ipynb", "classification.ipynb"):
        path = ROOT / "notebooks" / name
        if not path.exists():
            continue
        nb = nbformat.read(path, as_version=4)
        # Count placeholder cells (a "### ... TEAM ANALYSIS REQUIRED" heading), not cells that
        # merely mention the phrase, such as the notebook's reading guide.
        heading = re.compile(r"^\s*>?\s*#{1,6}[^\n]*" + re.escape(config.TEAM_MARKER), re.M)
        n = sum(bool(heading.search(c.source)) for c in nb.cells if c.cell_type == "markdown")
        total += n
        add("Team sections", f"{name}: unwritten TEAM ANALYSIS REQUIRED cells",
            "PENDING" if n else "PASS", f"{n} remaining")
    for track, st in feature_engineering.status().items():
        add("Team sections", f"engineered feature registered and justified ({track}, rubric B3)",
            "PASS" if st["complete"] else "PENDING",
            f"{st['n_justified']} justified of {st['n_registered']} registered")
    return total


def check_git():
    if not (ROOT / ".git").exists():
        add("Repository", "git repository initialised", "FAIL", "no .git folder")
        return
    add("Repository", "git repository initialised", "PASS")
    probe = [rel(c.path) for c in config.CONTRACTS.values()]
    try:
        out = subprocess.run(["git", "check-ignore", *probe], cwd=ROOT, capture_output=True, text=True)
        ignored = set(out.stdout.splitlines())   # one path per line; file names may contain spaces
        not_ignored = [p for p in probe if p not in ignored]
        add("Repository", "raw data is git-ignored (licensing, see docs/dataset_sources.md)",
            "FAIL" if not_ignored else "PASS", ", ".join(not_ignored) or "all raw CSVs ignored")
    except FileNotFoundError:
        add("Repository", "raw data is git-ignored", "PENDING", "git executable not found")


# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strict", action="store_true", help="treat PENDING as failure")
    parser.add_argument("--json", type=Path, help="also write the report as JSON")
    args = parser.parse_args()

    for fn in (check_structure, check_environment, check_raw_data, check_exclusions,
               check_catalogue, check_scope_guard, check_notebooks, check_models,
               check_results, check_team_sections, check_git):
        try:
            fn()
        except Exception as exc:   # a crashing check is itself a failure
            add(fn.__name__.replace("check_", "").title(), "check crashed", "FAIL", f"{type(exc).__name__}: {exc}")

    group = None
    for c in CHECKS:
        if c.group != group:
            group = c.group
            print(f"\n[{group}]")
        print(f"  {c.status:<8} {c.name}" + (f"  -- {c.detail}" if c.detail else ""))

    n = {s: sum(c.status == s for c in CHECKS) for s in ("PASS", "FAIL", "PENDING")}
    print(f"\nSUMMARY: {n['PASS']} pass, {n['FAIL']} fail, {n['PENDING']} pending")
    if n["FAIL"]:
        print("STATUS: FAILED - fix the FAIL items above.")
    elif n["PENDING"]:
        print("STATUS: NOT SUBMISSION-READY - mechanical checks pass, but PENDING items remain "
              "(later build phases and/or team-authored work).")
    else:
        print("STATUS: SUBMISSION-READY")
    if args.json:
        args.json.write_text(json.dumps([c.__dict__ for c in CHECKS], indent=2), encoding="utf-8")
    return 1 if n["FAIL"] or (args.strict and n["PENDING"]) else 0


if __name__ == "__main__":
    sys.exit(main())
