"""Physiology a correct run must show, checked on five simulated days.

The last tests are strict expected failures: each encodes what the papers
require where the code disagrees (`docs/research/2026-10-19-equation-audit.md`).
They fail today, and the pull request that fixes the discrepancy turns them
into ordinary tests; strict mode makes an unexpected pass fail the suite.
"""

import numpy as np
import pytest

from arousal_dynamics import ArousalParameters, Trajectory
from arousal_dynamics.models import arousal as eq

DAY = 86400.0
# odeint's default absolute tolerance. A state that is nonnegative in exact
# arithmetic can sit this far below zero once it has decayed to nothing.
SOLVER_ATOL = 1.5e-8


def after_first_day(run: Trajectory) -> np.ndarray:
    """Mask of time points from the second day on, past the start-up transient."""
    return run.t >= DAY


def test_every_value_is_finite(office_week: Trajectory) -> None:
    assert np.all(np.isfinite(office_week.states))
    for name, series in office_week.outputs.items():
        assert np.all(np.isfinite(series)), name


def test_bounded_states_stay_in_range(office_week: Trajectory) -> None:
    p = ArousalParameters()
    assert np.all((office_week["P"] >= 0) & (office_week["P"] <= 1))
    theta_l = office_week["Theta_L"]
    assert np.all((theta_l >= p.nu_la - SOLVER_ATOL) & (theta_l <= SOLVER_ATOL))
    assert np.all(office_week["A"] >= -SOLVER_ATOL)
    # With audit D1, synthesis is rarely on and plasma melatonin decays to
    # about 1e-24 between bursts, so this checks sign, not presence.
    assert np.all(office_week["rho_b"] >= -SOLVER_ATOL)


def test_awake_through_the_forced_wake_window(office_week: Trajectory) -> None:
    clock = office_week.t_hours % 24
    window = after_first_day(office_week) & (clock >= 7.5) & (clock <= 23.0)
    assert np.all(office_week["S"][window] == 1)


def test_asleep_most_of_the_night(office_week: Trajectory) -> None:
    clock = office_week.t_hours % 24
    night = after_first_day(office_week) & ((clock >= 23.5) | (clock < 6.0))
    asleep = office_week["S"][night] == 0
    assert asleep.mean() > 0.9


def _at(run: Trajectory, name: str, hour: float) -> float:
    """`run[name]` at the output point nearest `hour` hours from the start."""
    return float(run[name][np.argmin(np.abs(run.t_hours - hour))])


def test_sleep_pressure_builds_by_day_and_clears_by_night(office_week: Trajectory) -> None:
    for day in range(1, 5):
        waking, bed = 24 * day + 7.0, 24 * day + 23.0
        assert _at(office_week, "H", bed) > _at(office_week, "H", waking)
        if day < 4:
            assert _at(office_week, "H", waking + 24) < _at(office_week, "H", bed)


def _rising_crossings_h(run: Trajectory) -> np.ndarray:
    """Times, in hours, at which X crosses zero upward, linearly interpolated."""
    X, t = run["X"], run.t
    i = np.flatnonzero((X[:-1] < 0) & (X[1:] >= 0))
    return (t[i] - X[i] * (t[i + 1] - t[i]) / (X[i + 1] - X[i])) / 3600


@pytest.mark.slow
def test_circadian_rhythm_entrains_to_the_day(office_fortnight: Trajectory) -> None:
    # From the default initial state the oscillator takes weeks to entrain: the
    # first period is 23.4 h, and each later one closes part of the gap to the
    # 24 h day. Measured when written: 23.37 h on day one, 23.98 h on day 14,
    # 24.00 h by day 25.
    periods_h = np.diff(_rising_crossings_h(office_fortnight))
    assert periods_h.size >= 12
    assert np.all(np.diff(periods_h) > 0)
    assert abs(periods_h[-1] - 24.0) < 0.02


def test_kss_stays_on_the_scale(office_week: Trajectory) -> None:
    kss = office_week["KSS"][office_week["S"] == 1]
    assert np.all((kss >= 1.0) & (kss <= 9.0))


def test_light_lowers_sleepiness(office_week: Trajectory) -> None:
    # T20: bright light during the day pushes Theta_L, and with it KSS, down.
    clock = office_week.t_hours % 24
    bright = after_first_day(office_week) & (clock >= 9) & (clock < 17)
    dim = after_first_day(office_week) & (clock >= 19) & (clock < 22)
    assert office_week["Theta_L"][bright].mean() < office_week["Theta_L"][dim].mean()


@pytest.mark.xfail(
    strict=True,
    reason="audit D1: two-quadrant arctan gives two short synthesis windows a day",
)
def test_melatonin_is_synthesized_for_one_long_window_each_night(office_week: Trajectory) -> None:
    p = ArousalParameters()
    on = eq.melatonin_synthesis(office_week["phi"], p) == 1
    hours_per_day = on[after_first_day(office_week)].mean() * 24
    assert 6.0 <= hours_per_day <= 10.0


@pytest.mark.xfail(strict=True, reason="audit D2: logistic form gives r = 0.509 in darkness")
def test_darkness_does_not_suppress_melatonin() -> None:
    assert eq.melatonin_suppression(0.0, ArousalParameters()) == pytest.approx(1.0)
