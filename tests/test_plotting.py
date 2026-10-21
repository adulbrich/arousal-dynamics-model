"""The plotting helpers draw what they promise and fail clearly without matplotlib."""

import sys
from collections.abc import Iterator

import matplotlib.pyplot as plt
import pytest

from arousal_dynamics import Trajectory
from arousal_dynamics.plotting import plot_light, plot_trajectory


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def test_light_plot_marks_every_wake_and_bed_time(office_week: Trajectory) -> None:
    fig = plot_light(office_week)
    (ax,) = fig.axes
    days = 6  # five days of output plus the closing midnight
    # one wake and one bed line per day, plus a midnight line for each later day
    assert len(ax.lines) == 1 + 2 * days + (days - 1)
    legend = ax.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == ["Wake time", "Bed time"]


def test_trajectory_plot_has_one_panel_per_name(office_week: Trajectory) -> None:
    fig = plot_trajectory(office_week, ["KSS", "H", "rho_b"])
    assert [ax.get_ylabel() for ax in fig.axes] == ["KSS", "H", "rho_b"]


def test_trajectory_plot_defaults_to_everything(office_week: Trajectory) -> None:
    assert len(plot_trajectory(office_week).axes) == len(office_week.names)


def test_clear_error_without_matplotlib(
    office_week: Trajectory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match=r"arousal-dynamics\[plot\]"):
        plot_light(office_week)
