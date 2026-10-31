"""Background resolution: projection mode never draws class regions."""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from explviz.config import DatasetSpec, Explanation, Panel, resolve_background
from explviz.plot.panel import draw_panel

SPEC = DatasetSpec(
    name="toy", var_names=["x1", "x2", "x3"], axis_labels=["A", "B", "C"],
    bounds=[(0.0, 1.0)] * 3,
)
BOX = "(and (<= x1 (/ 8 10)) (>= x1 (/ 2 10)) (<= x2 (/ 8 10)) (>= x2 (/ 2 10)))"


def a_panel(**kw):
    kw.setdefault("point", np.array([0.5, 0.5, 0.5]))
    kw.setdefault("axes", (0, 1))
    kw.setdefault("explanations", [Explanation(BOX, "box", "box")])
    return Panel(**kw)


# ------------------------------------------------------------- pure resolution

@pytest.mark.parametrize("background", ["auto", "contour", "scatter", "none"])
def test_projection_never_draws_a_background(background):
    assert resolve_background("projection", background) == "none"


@pytest.mark.parametrize("background,expected", [
    ("auto", "contour"), ("contour", "contour"),
    ("scatter", "scatter"), ("none", "none"),
])
def test_snapshot_honours_the_request(background, expected):
    assert resolve_background("snapshot", background) == expected


def test_panel_default_is_auto():
    assert a_panel().background == "auto"
    assert a_panel(mode="snapshot").effective_background() == "contour"
    assert a_panel(mode="projection").effective_background() == "none"


# ------------------------------------------------------------------ drawing

def test_projection_needs_no_classifier():
    """The whole point: no model required when there is no background."""
    fig, ax = plt.subplots()
    draw_panel(ax, a_panel(mode="projection"), SPEC, clf=None)
    plt.close(fig)


def test_snapshot_still_requires_a_classifier():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="needs a classifier"):
        draw_panel(ax, a_panel(mode="snapshot"), SPEC, clf=None)
    plt.close(fig)


def test_explicit_background_in_projection_warns():
    fig, ax = plt.subplots()
    with pytest.warns(RuntimeWarning, match="ignored in projection mode"):
        draw_panel(ax, a_panel(mode="projection", background="contour"), SPEC, clf=None)
    plt.close(fig)


@pytest.mark.parametrize("background", ["auto", "none"])
def test_projection_does_not_warn_when_nothing_was_asked_for(background):
    fig, ax = plt.subplots()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        draw_panel(ax, a_panel(mode="projection", background=background), SPEC, clf=None)
    plt.close(fig)


def test_projection_draws_no_class_legend_entries():
    fig, ax = plt.subplots()
    draw_panel(ax, a_panel(mode="projection"), SPEC, clf=None)
    labels = ax.get_legend_handles_labels()[1]
    assert "High risk" not in labels and "Low risk" not in labels
    assert "box" in labels          # the explanation itself is still drawn
    plt.close(fig)
