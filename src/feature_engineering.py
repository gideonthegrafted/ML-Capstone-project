"""Feature-engineering hook - SHIPS EMPTY BY DESIGN.

Rubric B3 requires at least one engineered feature "with a written
justification of why it may improve model performance". Course guideline 7.5
requires feature-engineering decisions to be the team's own work, so this
module provides only the machinery. No feature is registered.

HOW THE TEAM REGISTERS A FEATURE
--------------------------------
Add a ``register_feature(...)`` call in the TEAM REGISTRATIONS section at the
bottom of this file, e.g.

    register_feature(RegisteredFeature(
        track="regression",
        name="<short_name>",
        new_columns=("<new column>",),
        function=<a function: DataFrame -> DataFrame with exactly new_columns>,
        justification="<your own reasoning, written by the team>",
        author="<student name>",
    ))

Rules for ``function``:
* It must be ROW-WISE: each output value may depend only on the same row's
  inputs. It must not compute means, quantiles or anything else across rows,
  because that would learn from the test set (leakage). A feature that needs
  fitted statistics must be written as a proper sklearn transformer instead.
* It must return a DataFrame with exactly ``new_columns`` and the same index.

The notebooks then show the with/without comparison automatically, and
``status()`` reports rubric item B3 as complete only when a registered feature
has a non-empty justification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TRACKS = ("regression", "classification")


@dataclass(frozen=True)
class RegisteredFeature:
    track: str
    name: str
    new_columns: tuple[str, ...]
    function: Callable[[pd.DataFrame], pd.DataFrame]
    justification: str
    author: str


_REGISTRY: dict[str, list[RegisteredFeature]] = {t: [] for t in TRACKS}


def register_feature(feature: RegisteredFeature) -> None:
    if feature.track not in TRACKS:
        raise ValueError(f"track must be one of {TRACKS}, got {feature.track!r}")
    if any(f.name == feature.name for f in _REGISTRY[feature.track]):
        raise ValueError(f"feature {feature.name!r} already registered for {feature.track}")
    _REGISTRY[feature.track].append(feature)


def registered_features(track: str) -> list[RegisteredFeature]:
    return list(_REGISTRY[track])


def status() -> dict:
    """Machine-readable completion state per track, used by validation."""
    out = {}
    for track in TRACKS:
        feats = _REGISTRY[track]
        justified = [f for f in feats if f.justification.strip() and f.author.strip()]
        out[track] = {
            "n_registered": len(feats),
            "n_justified": len(justified),
            "complete": len(justified) >= 1,
            "features": [f.name for f in feats],
        }
    return out


class AddRegisteredFeatures(BaseEstimator, TransformerMixin):
    """Pipeline step that appends a track's registered row-wise features.

    With nothing registered it passes the frame through unchanged, so the
    baseline pipeline runs before the team has added a feature.
    """

    def __init__(self, track: str):
        self.track = track

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feat in _REGISTRY[self.track]:
            new = feat.function(X)
            if tuple(new.columns) != tuple(feat.new_columns):
                raise ValueError(
                    f"{feat.name}: function returned columns {list(new.columns)}, "
                    f"declared {list(feat.new_columns)}"
                )
            if not new.index.equals(X.index):
                raise ValueError(f"{feat.name}: function changed the row index")
            X = pd.concat([X, new], axis=1)
        return X


# ==========================================================================
# TEAM REGISTRATIONS  (TEAM ANALYSIS REQUIRED - rubric B3)
# Intentionally empty. Register your feature(s) below, with your own
# written justification. Do not ask an AI tool to write the justification.
# ==========================================================================
