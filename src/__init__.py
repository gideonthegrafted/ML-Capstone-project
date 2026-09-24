"""Shared support code for the 23CSE301 ML Capstone (Review 1).

The notebooks are the primary deliverable: every model is built, trained,
tuned and evaluated visibly there. This package holds reusable plumbing only
(paths, contracts, loading/auditing, unfitted preprocessing builders, metric
conventions and plot styling), so notebooks and scripts cannot drift apart.
"""
import inspect


def show_source(obj) -> None:
    """Print the source code of a helper so it is visible inside the notebook."""
    print(inspect.getsource(obj))
