"""Matplotlib views of a trajectory. Needs the `plot` extra: `arousal-dynamics[plot]`."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from arousal_dynamics.inputs import WakeSchedule
from arousal_dynamics.simulation import Trajectory

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _pyplot():
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as error:
        msg = "plotting needs matplotlib: install arousal-dynamics[plot]"
        raise ImportError(msg) from error
    return plt


def plot_light(trajectory: Trajectory, schedule: WakeSchedule | None = None) -> "Figure":
    """Plot the light the model saw, with the forced-wake window marked each day.

    This is the plot the original `model_run(debug=True)` drew before
    integrating. A run without a forced-wake window gets no window marks.
    """
    plt = _pyplot()
    schedule = schedule if schedule is not None else trajectory.inputs.schedule
    fig, ax = plt.subplots(figsize=(12, 6))
    hours = trajectory.t_hours
    ax.plot(hours, trajectory["light"])
    ax.set(
        title="Light input",
        xlabel="Time (h)",
        ylabel="Melanopic irradiance (W/m$^2$)",
    )
    ax.grid(visible=True)
    days = int(hours[-1] // 24) + 1
    for day in range(days):
        offset = 24 * day
        if day:
            ax.axvline(offset, color="r", linestyle="--", alpha=0.3)
        if schedule is None:
            continue
        ax.axvline(
            offset + schedule.waketime,
            color="g",
            linestyle=":",
            label="Wake time" if not day else None,
        )
        ax.axvline(
            offset + schedule.bedtime,
            color="b",
            linestyle=":",
            label="Bed time" if not day else None,
        )
    if schedule is not None:
        ax.legend()
    return fig


def plot_trajectory(trajectory: Trajectory, names: Sequence[str] | None = None) -> "Figure":
    """One panel per state or output, sharing the time axis in hours."""
    plt = _pyplot()
    names = list(names) if names is not None else list(trajectory.names)
    fig, axes = plt.subplots(
        len(names), 1, sharex=True, figsize=(10, 1.8 * len(names)), squeeze=False
    )
    for ax, name in zip(axes[:, 0], names, strict=True):
        ax.plot(trajectory.t_hours, trajectory[name])
        ax.set_ylabel(name)
        ax.grid(visible=True)
    axes[-1, 0].set_xlabel("Time (h)")
    fig.tight_layout()
    return fig
