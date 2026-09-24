"""The feature-engineering hook ships empty; plots enforce labelling rules."""
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src import config, feature_engineering as fe
from src import plotting


def test_feature_hook_ships_empty_and_incomplete():
    st = fe.status()
    for track in fe.TRACKS:
        assert st[track]["n_registered"] == 0
        assert st[track]["complete"] is False


def test_feature_hook_machinery_works_and_requires_justification(monkeypatch):
    monkeypatch.setattr(fe, "_REGISTRY", {t: [] for t in fe.TRACKS})
    fe.register_feature(fe.RegisteredFeature(
        track="regression", name="demo", new_columns=("a_x2",),
        function=lambda X: pd.DataFrame({"a_x2": X["a"] * 2}, index=X.index),
        justification="", author="test",
    ))
    assert fe.status()["regression"]["complete"] is False        # empty justification
    out = fe.AddRegisteredFeatures("regression").fit_transform(pd.DataFrame({"a": [1, 2]}))
    assert out["a_x2"].tolist() == [2, 4]


def test_passthrough_when_nothing_registered():
    X = pd.DataFrame({"a": [1, 2]})
    pd.testing.assert_frame_equal(fe.AddRegisteredFeatures("regression").fit_transform(X), X)


def test_finalize_rejects_unlabelled_plots_and_saves(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    with pytest.raises(ValueError):
        plotting.finalize(fig, "x")                 # no title / labels
    ax.set_title("t"); ax.set_xlabel("x"); ax.set_ylabel("y")
    path = plotting.finalize(fig, "ok")
    assert path.exists() and path.parent == tmp_path
    plt.close(fig)
