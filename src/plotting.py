"""House style and reusable figures.

Course guideline 7.3: every plot has a title, labelled axes and a legend where
applicable; tight layout; colourblind-friendly palette. ``finalize`` enforces
this - it raises if a title or axis label is missing - and saves the figure
to results/figures.

No backend is forced here, so figures display inline in Jupyter. Scripts that
run headless set the environment variable MPLBACKEND=Agg.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config

PALETTE = "colorblind"


def set_style() -> None:
    """Apply the project plotting style (call once per notebook)."""
    sns.set_theme(style="whitegrid", palette=PALETTE, context="notebook")
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
    })


def _check_axes(ax, require_legend: bool) -> None:
    if not ax.get_title():
        raise ValueError("every plot needs a title (guideline 7.3)")
    if not ax.get_xlabel() or not ax.get_ylabel():
        raise ValueError(f"axes '{ax.get_title()}' needs both axis labels (guideline 7.3)")
    if require_legend and ax.get_legend() is None:
        raise ValueError(f"axes '{ax.get_title()}' needs a legend")


def finalize(fig, filename: str | None, axes=None, require_legend: bool = False,
             check: bool = True):
    """Validate labelling, apply tight layout and save to results/figures.

    ``axes`` defaults to every axes in the figure that carries data (colorbars
    are skipped). ``filename`` without extension; dev runs get a suffix.
    Returns the saved path (or None when ``filename`` is None).
    """
    if check:
        targets = axes if axes is not None else [a for a in fig.axes if a.get_label() != "<colorbar>"]
        for ax in np.atleast_1d(targets):
            _check_axes(ax, require_legend)
    fig.tight_layout()
    if filename is None:
        return None
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{filename}{config.OUTPUT_SUFFIX}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


# --------------------------------------------------------------------------
# Reusable figures (all return the figure; callers finalize/save)
# --------------------------------------------------------------------------
def correlation_heatmap(df: pd.DataFrame, title: str):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(1.1 * len(corr) + 3, 0.9 * len(corr) + 2))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1,
                square=True, cbar_kws={"label": "Pearson r"}, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Feature")
    return fig


def bar_comparison(table: pd.DataFrame, metric: str, title: str, xlabel: str | None = None):
    """Horizontal bar chart of one metric across models (best at the top)."""
    data = table[metric].sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(data) + 1.5))
    ax.barh(data.index, data.values, color=sns.color_palette(PALETTE)[0])
    for y, v in enumerate(data.values):
        ax.text(v, y, f" {v:.4f}", va="center", fontsize=9)
    ax.set_title(title)
    ax.set_xlabel(xlabel or metric)
    ax.set_ylabel("Model")
    return fig


def predicted_vs_actual(y_true, y_pred, title: str, units: str = ""):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=10, alpha=0.4, label="test rows")
    lo, hi = float(min(np.min(y_true), np.min(y_pred))), float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1, label="perfect prediction")
    ax.set_title(title)
    ax.set_xlabel(f"Actual {units}".strip())
    ax.set_ylabel(f"Predicted {units}".strip())
    ax.legend()
    return fig


def residual_plot(y_true, y_pred, title: str, units: str = ""):
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(y_pred, residuals, s=10, alpha=0.4, label="test rows")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1, label="zero residual")
    axes[0].set_title(f"{title}: residuals vs predicted")
    axes[0].set_xlabel(f"Predicted {units}".strip())
    axes[0].set_ylabel(f"Residual (actual - predicted) {units}".strip())
    axes[0].legend()
    sns.histplot(residuals, bins=40, kde=True, ax=axes[1])
    axes[1].set_title(f"{title}: residual distribution")
    axes[1].set_xlabel(f"Residual {units}".strip())
    axes[1].set_ylabel("Count")
    return fig


def feature_importance(names, importances, title: str, top_n: int = 20,
                       xlabel: str = "Importance"):
    s = pd.Series(importances, index=names).sort_values(ascending=False).head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(s) + 1.5))
    ax.barh(s.index, s.values, color=sns.color_palette(PALETTE)[2])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Feature")
    return fig


def confusion_matrix_plot(cm: pd.DataFrame, title: str, ax=None):
    """Heatmap of a labelled confusion matrix from evaluation.labelled_confusion_matrix."""
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(4.8, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=[c.replace("predicted: ", "") for c in cm.columns],
                yticklabels=[i.replace("actual: ", "") for i in cm.index])
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    return ax.figure


def roc_curves(curves: dict, title: str, pos_label: str = config.POS_LABEL):
    """Overlay ROC curves. ``curves`` maps model name -> (fpr, tpr, auc)."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="chance")
    ax.set_title(title)
    ax.set_xlabel(f"False positive rate (positive class = {pos_label})")
    ax.set_ylabel(f"True positive rate (recall of {pos_label})")
    ax.legend(loc="lower right", fontsize=8)
    return fig
