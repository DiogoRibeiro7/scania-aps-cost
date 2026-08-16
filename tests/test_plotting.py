"""Tests for the notebook chart helpers.

These assert the encoding rules the figures rely on, not pixel output: a single
hue for magnitude, an accent against a de-emphasis grey for emphasis, and a
diverging pair that does not borrow the reserved status colours.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip(
    "matplotlib", reason="requires the optional 'notebooks' dependency group"
)
matplotlib.use("Agg")

from scania_aps import plotting  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures() -> object:
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


def test_categorical_slots_are_distinct_and_capped() -> None:
    """Eight slots, no duplicates. A ninth hue would be indistinguishable."""

    assert len(plotting.SERIES) == 8
    assert len(set(plotting.SERIES)) == 8
    assert all(c.startswith("#") and len(c) == 7 for c in plotting.SERIES)


def test_status_colours_are_not_reused_as_series() -> None:
    """A status colour must never impersonate a series, or the reverse."""

    assert plotting.STATUS_GOOD not in plotting.SERIES
    assert plotting.STATUS_CRITICAL not in plotting.SERIES


def test_sequential_ramp_is_monotonically_darker() -> None:
    """Magnitude needs one hue, light to dark; a rainbow would be unreadable."""

    def luminance(hexcolor: str) -> float:
        r, g, b = (int(hexcolor[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    lums = [luminance(c) for c in plotting.SEQUENTIAL]
    assert lums == sorted(lums, reverse=True), "ramp must darken monotonically"


def test_apply_house_style_sets_shared_rcparams() -> None:
    import matplotlib.pyplot as plt

    plotting.apply_house_style()

    assert plt.rcParams["axes.facecolor"] == plotting.SURFACE
    assert plt.rcParams["axes.spines.top"] is False
    # A dashed grid reads as a threshold; the grid is not a threshold.
    assert plt.rcParams["grid.linestyle"] == "-"


def test_cost_curve_marks_the_optimum_and_the_bayes_line() -> None:
    plotting.apply_house_style()
    thresholds = np.linspace(0.001, 0.5, 200)
    costs = 1000.0 / (thresholds + 0.05) + 5000 * thresholds

    fig, ax = plotting.cost_curve(
        thresholds, costs, optimal_threshold=0.02, bayes_threshold=10 / 510
    )

    assert ax.get_xlabel() == "Decision threshold"
    assert ax.collections, "the optimum should be marked with a point"
    assert any("Bayes" in t.get_text() for t in ax.texts)


def test_emphasis_bars_highlight_the_cheapest_by_default() -> None:
    plotting.apply_house_style()
    fig, ax = plotting.emphasis_bars(
        ["expensive", "cheap", "middling"], [300.0, 100.0, 200.0], title="Cost"
    )

    colors = [patch.get_facecolor() for patch in ax.patches]
    accent = matplotlib.colors.to_rgba(plotting.SERIES[0])
    muted = matplotlib.colors.to_rgba(plotting.MUTED_MARK)

    assert colors.count(accent) == 1, "exactly one bar carries the accent"
    assert colors.count(muted) == 2
    # Sorted ascending and drawn top-down, so the cheapest is the first bar.
    assert colors[0] == accent


def test_emphasis_bars_can_highlight_the_maximum() -> None:
    plotting.apply_house_style()
    fig, ax = plotting.emphasis_bars(["a", "b", "c"], [1.0, 3.0, 2.0], title="t", highlight="max")

    accent = matplotlib.colors.to_rgba(plotting.SERIES[0])
    assert [p.get_facecolor() for p in ax.patches][-1] == accent


def test_diverging_bars_default_to_the_diverging_pair_not_status() -> None:
    """Direction is not virtue: a signed quantity gets blue/red, not green/red."""

    plotting.apply_house_style()
    fig, ax = plotting.diverging_bars(["up", "down"], [5.0, -5.0], title="t")

    colors = {matplotlib.colors.to_hex(p.get_facecolor()) for p in ax.patches}
    assert colors == {plotting.DIVERGING_HIGH.lower(), plotting.DIVERGING_LOW.lower()}
    assert plotting.STATUS_GOOD.lower() not in colors


def test_diverging_bars_use_status_colours_only_when_asked() -> None:
    plotting.apply_house_style()
    fig, ax = plotting.diverging_bars(["worse", "better"], [5.0, -5.0], title="t", semantic=True)

    colors = {matplotlib.colors.to_hex(p.get_facecolor()) for p in ax.patches}
    assert colors == {plotting.STATUS_CRITICAL.lower(), plotting.STATUS_GOOD.lower()}


def test_magnitude_bars_use_a_single_hue() -> None:
    """One series is one colour; a per-bar ramp would double-encode length."""

    plotting.apply_house_style()
    fig, ax = plotting.magnitude_bars(
        [f"f{i}" for i in range(6)], [6.0, 5.0, 4.0, 3.0, 2.0, 1.0], title="t"
    )

    colors = {matplotlib.colors.to_hex(p.get_facecolor()) for p in ax.patches}
    assert colors == {plotting.SERIES[0].lower()}


def test_reliability_diagram_draws_the_diagonal_and_one_line_per_method() -> None:
    plotting.apply_house_style()
    predicted = np.linspace(0.05, 0.95, 8)
    fig, ax = plotting.reliability_diagram(
        {
            "uncalibrated": (predicted, predicted**1.5),
            "isotonic": (predicted, predicted**1.02),
        }
    )

    # One diagonal reference plus one line per method.
    assert len(ax.lines) == 3
    assert ax.get_xlim() == (0.0, 1.0)
    assert ax.get_legend() is not None


def test_training_curves_label_each_series_at_its_endpoint() -> None:
    plotting.apply_house_style()
    history = [{"epoch": float(e), "val_loss": 1.0 / (1 + e)} for e in range(6)]

    fig, ax = plotting.training_curves({"adam": history, "sgd": history})

    labels = {t.get_text() for t in ax.texts}
    assert {"adam", "sgd"} <= labels


def test_series_lines_assign_slots_in_fixed_order() -> None:
    """Colour follows the entity, so slot order must not depend on the data."""

    plotting.apply_house_style()
    x = np.linspace(0.01, 10, 5)
    fig, ax = plotting.series_lines(
        x,
        {"l1": x * 2, "l2": x * 3, "elasticnet": x * 4},
        title="t",
        xlabel="x",
        ylabel="y",
        log_x=True,
        mark_minimum=True,
    )

    drawn = [matplotlib.colors.to_hex(line.get_color()) for line in ax.lines]
    assert drawn == [c.lower() for c in plotting.SERIES[:3]]
    assert ax.get_xscale() == "log"


def test_a_single_series_needs_no_legend_box() -> None:
    """The title names one series; a legend for it is noise."""

    plotting.apply_house_style()
    x = np.linspace(0, 1, 5)
    fig, ax = plotting.series_lines(x, {"only": x}, title="t", xlabel="x", ylabel="y")

    assert ax.get_legend() is None


def test_subtitle_does_not_collide_with_the_title() -> None:
    """The subtitle sits below the title, not on top of it."""

    plotting.apply_house_style()
    fig, ax = plotting.magnitude_bars(["a", "b"], [1.0, 2.0], title="Title", subtitle="Subtitle")

    subtitle = next(t for t in ax.texts if t.get_text() == "Subtitle")
    fig.canvas.draw()
    title_y = ax.title.get_window_extent().y0
    subtitle_y = subtitle.get_window_extent().y1
    assert subtitle_y <= title_y, "subtitle overlaps the title"
