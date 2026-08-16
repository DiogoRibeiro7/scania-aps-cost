"""House chart style and reusable figures for the experiment notebooks.

Matplotlib lives in the optional ``notebooks`` group, so it is imported inside
the functions that need it and never at module import time. Importing this
module is therefore always safe; calling a plotting function without the group
installed raises a clear error naming what to install.

Figures are committed into the notebooks and rendered on GitHub, where the page
background follows the reader's theme. Every figure therefore paints an explicit
light surface and reads as a self-contained light card in either theme, rather
than relying on a transparent background whose ink would vanish in dark mode.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - import only for annotations
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Categorical slots, assigned in fixed order and never cycled. Validated for
# colour-vision deficiency as an ordered set: worst adjacent pair is yellow to
# aqua at CVD deltaE 9.1, above the 8 threshold.
SERIES: tuple[str, ...] = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Reserved for state, never for series identity.
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"

# Diverging poles for signed quantities: warm against cool, so the two ends read
# as opposite while the zero line reads as nothing. Two cool hues would not.
DIVERGING_LOW = "#2a78d6"
DIVERGING_HIGH = "#e34948"

# One hue, light to dark, for magnitude.
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")

# De-emphasis grey for the emphasis form: one series in colour, the rest recede.
MUTED_MARK = "#d6d5cf"


def _pyplot() -> object:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Install the optional 'notebooks' dependency group to draw figures."
        ) from exc
    return plt


def apply_house_style() -> None:
    """Set the shared rcParams. Call once near the top of a notebook."""

    plt = _pyplot()
    plt.rcParams.update(  # type: ignore[attr-defined]
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 120,
            "savefig.dpi": 120,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",  # never dashed; a grid is not a threshold
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "figure.autolayout": True,
        }
    )


def _finish(ax: Axes, *, title: str, subtitle: str | None, xlabel: str, ylabel: str) -> None:
    if subtitle:
        # Reserve room for both lines, or the subtitle prints over the title.
        ax.set_title(title, pad=32)
        ax.text(
            0.0,
            1.015,
            subtitle,
            transform=ax.transAxes,
            fontsize=9,
            color=INK_SECONDARY,
            va="bottom",
        )
    else:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def cost_curve(
    thresholds: NDArray[np.float64],
    costs: NDArray[np.float64],
    *,
    optimal_threshold: float | None = None,
    bayes_threshold: float | None = None,
    title: str = "Maintenance cost against decision threshold",
    subtitle: str | None = None,
    figsize: tuple[float, float] = (7.5, 4.2),
) -> tuple[Figure, Axes]:
    """Plot cost as a function of the operating threshold.

    This is the project's signature figure: it shows why 0.5 is not a defensible
    default, by making the cost penalty of the wrong threshold visible.
    """

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=figsize)  # type: ignore[attr-defined]

    ax.plot(thresholds, costs, color=SERIES[0], zorder=3)

    if optimal_threshold is not None:
        best_cost = float(np.interp(optimal_threshold, thresholds, costs))
        ax.scatter(
            [optimal_threshold],
            [best_cost],
            s=90,
            color=SERIES[0],
            edgecolor=SURFACE,
            linewidth=2,  # 2px surface ring on the overlapping mark
            zorder=5,
        )
        ax.annotate(
            f"cost-optimal  τ*={optimal_threshold:.4f}\ncost {best_cost:,.0f}",
            xy=(optimal_threshold, best_cost),
            xytext=(12, 18),
            textcoords="offset points",
            fontsize=9,
            color=INK,
            ha="left",
        )

    if bayes_threshold is not None:
        ax.axvline(bayes_threshold, color=INK_MUTED, linewidth=1.0, zorder=2)
        ax.annotate(
            f"Bayes τ={bayes_threshold:.4f}",
            xy=(bayes_threshold, ax.get_ylim()[1]),
            xytext=(6, -14),
            textcoords="offset points",
            fontsize=9,
            color=INK_SECONDARY,
            ha="left",
            va="top",
        )

    _finish(
        ax,
        title=title,
        subtitle=subtitle,
        xlabel="Decision threshold",
        ylabel="Maintenance cost  (10·FP + 500·FN)",
    )
    return fig, ax


def emphasis_bars(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    highlight: Literal["min", "max"] | int = "min",
    title: str,
    subtitle: str | None = None,
    xlabel: str = "Maintenance cost",
    value_format: str = "{:,.0f}",
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Horizontal bars with one highlighted and the rest receding.

    When the story is "which one wins", eight categorical hues bury the answer.
    One accent plus a de-emphasis grey states it.
    """

    plt = _pyplot()
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]

    if highlight == "min":
        marked = 0
    elif highlight == "max":
        marked = len(values) - 1
    else:
        marked = int(np.where(order == highlight)[0][0])

    height = figsize or (7.5, max(2.4, 0.42 * len(labels) + 1.4))
    fig, ax = plt.subplots(figsize=height)  # type: ignore[attr-defined]

    colors = [SERIES[0] if i == marked else MUTED_MARK for i in range(len(values))]
    positions = np.arange(len(values))
    ax.barh(positions, values, color=colors, height=0.62, zorder=3)

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)

    span = max(values) if max(values) else 1.0
    for pos, value, index in zip(positions, values, range(len(values)), strict=True):
        ax.text(
            value + span * 0.012,
            pos,
            value_format.format(value),
            va="center",
            fontsize=9,
            color=INK if index == marked else INK_SECONDARY,
        )
    ax.set_xlim(0, span * 1.16)

    _finish(ax, title=title, subtitle=subtitle, xlabel=xlabel, ylabel="")
    return fig, ax


def diverging_bars(
    labels: Sequence[str],
    deltas: Sequence[float],
    *,
    title: str,
    subtitle: str | None = None,
    xlabel: str = "Change in maintenance cost vs the full model",
    semantic: bool = False,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Signed values around a neutral zero.

    The default palette is the diverging pair blue-to-red: two hues that read as
    opposite without asserting that either side is *better*. Most signed
    quantities here are directional, not good or bad -- "missing more often on
    failed trucks" has a sign but no virtue.

    Set ``semantic=True`` only where the sign genuinely means good or bad, such
    as an ablation where a cost increase is a regression. Status colours carry
    that meaning and must not be spent on plain identity or direction.
    """

    plt = _pyplot()
    size = figsize or (7.5, max(2.4, 0.42 * len(labels) + 1.4))
    fig, ax = plt.subplots(figsize=size)  # type: ignore[attr-defined]

    positive, negative = (
        (STATUS_CRITICAL, STATUS_GOOD) if semantic else (DIVERGING_HIGH, DIVERGING_LOW)
    )
    positions = np.arange(len(labels))
    colors = [positive if d > 0 else negative for d in deltas]
    ax.barh(positions, deltas, color=colors, height=0.62, zorder=3)
    ax.axvline(0.0, color=AXIS, linewidth=1.0, zorder=4)

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)

    span = max(abs(min(deltas)), abs(max(deltas))) or 1.0
    for pos, delta in zip(positions, deltas, strict=True):
        offset = span * 0.02
        ax.text(
            delta + (offset if delta >= 0 else -offset),
            pos,
            f"{delta:+,.0f}",
            va="center",
            ha="left" if delta >= 0 else "right",
            fontsize=9,
            color=INK_SECONDARY,
        )
    ax.set_xlim(-span * 1.28, span * 1.28)

    _finish(ax, title=title, subtitle=subtitle, xlabel=xlabel, ylabel="")
    return fig, ax


def reliability_diagram(
    curves: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    *,
    title: str = "Reliability of predicted probabilities",
    subtitle: str | None = None,
    figsize: tuple[float, float] = (5.6, 5.2),
) -> tuple[Figure, Axes]:
    """Predicted probability against observed frequency, per calibration method.

    ``curves`` maps a label to ``(mean_predicted, observed_fraction)``. Perfect
    calibration is the diagonal; the distance from it is the miscalibration.
    """

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=figsize)  # type: ignore[attr-defined]

    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1.0, zorder=2)
    ax.text(
        0.97,
        0.99,
        "perfectly calibrated",
        transform=ax.transAxes,
        fontsize=9,
        color=INK_MUTED,
        ha="right",
        va="top",
        rotation=0,
    )

    for index, (label, (predicted, observed)) in enumerate(curves.items()):
        ax.plot(
            predicted,
            observed,
            color=SERIES[index],
            marker="o",
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label=label,
            zorder=3 + index,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    _finish(
        ax,
        title=title,
        subtitle=subtitle,
        xlabel="Mean predicted probability",
        ylabel="Observed failure frequency",
    )
    return fig, ax


def training_curves(
    histories: dict[str, list[dict[str, float]]],
    *,
    metric: str = "val_loss",
    title: str = "Validation loss by optimizer",
    subtitle: str | None = None,
    figsize: tuple[float, float] = (7.5, 4.2),
) -> tuple[Figure, Axes]:
    """Epoch-level convergence, one line per configuration."""

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=figsize)  # type: ignore[attr-defined]

    for index, (label, history) in enumerate(histories.items()):
        epochs = [row["epoch"] for row in history]
        values = [row[metric] for row in history]
        color = SERIES[index % len(SERIES)]
        ax.plot(epochs, values, color=color, label=label, zorder=3 + index)
        # Direct-label the endpoint rather than every point.
        ax.annotate(
            label,
            xy=(epochs[-1], values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            color=color,
            va="center",
        )

    ax.legend(loc="upper right")
    _finish(
        ax,
        title=title,
        subtitle=subtitle,
        xlabel="Epoch",
        ylabel=metric.replace("_", " "),
    )
    return fig, ax


def magnitude_bars(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    subtitle: str | None = None,
    xlabel: str = "",
    value_format: str = "{:.3f}",
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Ranked magnitude in a single hue.

    One series means one colour. A per-bar ramp would double-encode length as
    hue and spend the only free channel on information the bar already carries.
    """

    plt = _pyplot()
    size = figsize or (7.5, max(2.4, 0.34 * len(labels) + 1.4))
    fig, ax = plt.subplots(figsize=size)  # type: ignore[attr-defined]

    positions = np.arange(len(labels))
    ax.barh(positions, values, color=SERIES[0], height=0.62, zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)

    span = max(values) if max(values) else 1.0
    for pos, value in zip(positions, values, strict=True):
        ax.text(
            value + span * 0.012,
            pos,
            value_format.format(value),
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )
    ax.set_xlim(0, span * 1.18)

    _finish(ax, title=title, subtitle=subtitle, xlabel=xlabel, ylabel="")
    return fig, ax


def series_lines(
    x: NDArray[np.float64],
    series: dict[str, NDArray[np.float64]],
    *,
    title: str,
    subtitle: str | None = None,
    xlabel: str,
    ylabel: str,
    log_x: bool = False,
    mark_minimum: bool = False,
    figsize: tuple[float, float] = (7.5, 4.2),
) -> tuple[Figure, Axes]:
    """Several comparable series on one axis, in fixed categorical order."""

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=figsize)  # type: ignore[attr-defined]

    for index, (label, values) in enumerate(series.items()):
        color = SERIES[index % len(SERIES)]
        ax.plot(x, values, color=color, label=label, marker="o", markersize=5, zorder=3 + index)
        if mark_minimum:
            best = int(np.argmin(values))
            ax.scatter(
                [x[best]],
                [values[best]],
                s=110,
                color=color,
                edgecolor=SURFACE,
                linewidth=2,
                zorder=9,
            )

    if log_x:
        ax.set_xscale("log")
    if len(series) > 1:
        ax.legend(loc="best")
    _finish(ax, title=title, subtitle=subtitle, xlabel=xlabel, ylabel=ylabel)
    return fig, ax
