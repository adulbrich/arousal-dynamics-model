"""Golden outputs, pinned on the legacy `models.py` at commit 55eeb3e.

These tests prove that a refactor changes no numbers. The arrays under
`tests/test_golden/` were written by the legacy code, bugs included: the
equation audit lists the known discrepancies with the papers, and each fix
lands in a pull request of its own that regenerates the affected files with

    uv run pytest tests/test_golden.py --force-regen

in a separate commit and says in the PR body which arrays moved and why.

Tolerances: the equation grids are pure arithmetic and must agree to a few
ulps. The trajectories come from an adaptive solver (LSODA via `odeint`), whose
step selection differs between SciPy releases: from SciPy 1.13 to 1.18 the
worst drift measured was 3e-6 of a variable's range over three days, largest
in V_v and V_m at the sleep/wake flips. Each trajectory is therefore compared
with an absolute tolerance of 1e-5 of its own range, which a changed equation,
constant or input handling exceeds by orders of magnitude. The sleep/wake
state is discrete and is compared exactly.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

import constants
import models

HOUR = 3600.0
DAY = 24 * HOUR
DAYS = 3
STEPS = DAYS * 144 + 1  # one output point every 10 minutes

# Melanopic EDI in lux to melanopic irradiance in W/m^2.
LUX_TO_W = 1.0 / constants.CONVERSION_FACTOR

STATE_NAMES = ("V_v", "V_m", "H", "X", "Y", "P", "Theta_L", "A", "rho_b")

EQUATION_TOLERANCE = {"rtol": 1e-12, "atol": 1e-15}
TRAJECTORY_RANGE_FRACTION = 1e-5


def _trajectory_tolerances(data: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Per-variable tolerance: a fixed fraction of the variable's range; t and S exact."""
    # The scale is the range, or the magnitude for a near-constant series whose
    # range is rounding noise (constant light interpolates to 0.1 +/- 1 ulp).
    tolerances = {
        key: {
            "rtol": 0.0,
            "atol": TRAJECTORY_RANGE_FRACTION
            * max(float(np.ptp(v)), float(np.abs(v).max()), 1e-12),
        }
        for key, v in data.items()
    }
    for exact in ("t", "S"):
        tolerances[exact] = {"rtol": 0.0, "atol": 0.0}
    return tolerances


def _each(f: Callable[..., float], *args: object) -> np.ndarray:
    """Call a legacy scalar function element by element, as the ODE calls it.

    Not through the legacy `*_v` wrappers: `np.vectorize` takes its output
    dtype from the first result, and `wake_effort` returns the integer 0 there,
    which truncates every later value to an integer.
    """
    arrays = np.broadcast_arrays(*(np.asarray(a) for a in args))
    return np.array(
        [f(*row) for row in zip(*(a.ravel() for a in arrays), strict=True)], dtype=float
    )


@dataclass(frozen=True)
class Scenario:
    """A light input and a sleep window, as the legacy `model_run` takes them."""

    name: str
    waketime: float
    bedtime: float
    hours: np.ndarray | None = None
    irradiance: np.ndarray | None = None
    interpolator: Callable[[float], float] | None = None


def _schedule(levels: list[tuple[float, float, float]], step_h: float) -> tuple[np.ndarray, ...]:
    """Sample a daily piecewise-constant schedule over the run plus one day.

    `levels` holds (start hour, end hour, W/m^2) rows covering [0, 24).
    """
    hours = np.arange(0.0, (DAYS + 1) * 24.0 + step_h, step_h)
    clock = hours % 24.0
    values = np.zeros_like(hours)
    for start, end, level in levels:
        values[(clock >= start) & (clock < end)] = level
    return hours, values


def _sun(t: float) -> float:
    """A smooth daylight curve: half-sine between 06:00 and 18:00, dark otherwise."""
    return 0.3 * max(0.0, float(np.sin(2.0 * np.pi * (t / DAY - 0.25))))


def _scenarios() -> list[Scenario]:
    constant_h, constant_e = _schedule([(0.0, 24.0, 0.1)], step_h=1.0)
    office_h, office_e = _schedule(
        [
            (0.0, 7.0, 0.0),
            (7.0, 8.0, 0.03),
            (8.0, 17.0, 0.25),
            (17.0, 23.0, 0.05),
            (23.0, 24.0, 0.0),
        ],
        step_h=1.0,
    )
    # The published 40 lux / 250 lux protocol in melanopic EDI, scaled by the
    # 0.6745812 melanopic ratio found in data/output.csv.
    dim, bright = 40 * 0.6745812 * LUX_TO_W, 250 * 0.6745812 * LUX_TO_W
    reference_h, reference_e = _schedule(
        [(0.0, 8.0, dim), (8.0, 20.0, bright), (20.0, 24.0, dim)],
        step_h=1.0 / 6.0,
    )
    return [
        Scenario("constant", 6, 22, constant_h, constant_e),
        Scenario("office", 7, 23, office_h, office_e),
        Scenario("reference_schedule", 8, 20, reference_h, reference_e),
        Scenario("interpolator", 6, 22, interpolator=_sun),
    ]


SCENARIOS = _scenarios()


def _run_legacy(scenario: Scenario) -> dict[str, np.ndarray]:
    if scenario.interpolator is None:
        assert scenario.hours is not None
        assert scenario.irradiance is not None
        sol, t = models.model_run(
            DAYS,
            STEPS,
            input_irradiance=scenario.irradiance,
            time_points=scenario.hours * HOUR,
            waketime=scenario.waketime,
            bedtime=scenario.bedtime,
        )
        # Rebuild the interpolator `model_run` made, to record the light the
        # model actually saw.
        from scipy.interpolate import interp1d  # noqa: PLC0415

        interpolator = interp1d(
            scenario.hours * HOUR,
            scenario.irradiance,
            kind="cubic",
            bounds_error=False,
            fill_value=(scenario.irradiance[0], scenario.irradiance[-1]),
        )
    else:
        interpolator = scenario.interpolator
        sol, t = models.model_run(
            DAYS,
            STEPS,
            interpolator=interpolator,
            waketime=scenario.waketime,
            bedtime=scenario.bedtime,
        )

    out = {name: sol[:, i] for i, name in enumerate(STATE_NAMES)}
    out["t"] = t
    out["light"] = np.array([models.irradiance(ti, interpolator) for ti in t], dtype=float)
    out["S"] = _each(models.state, out["V_m"])
    out["C"] = _each(models.circadian_drive, out["X"], out["Y"])
    out["KSS"] = _each(models.alertness_measure, out["C"], out["H"], out["Theta_L"])
    out["phi"] = _each(models.circadian_phase, out["X"], out["Y"])
    out["aMT6s"] = np.array([models.urinary_excretion_rate(out["rho_b"], t, ti) for ti in t])
    return out


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_trajectory(scenario: Scenario, ndarrays_regression: NDArraysRegressionFixture) -> None:
    data = _run_legacy(scenario)
    ndarrays_regression.check(
        data,
        basename=f"trajectory_{scenario.name}",
        tolerances=_trajectory_tolerances(data),
    )


def _grid(lo: float, hi: float, n: int = 41) -> np.ndarray:
    return np.linspace(lo, hi, n)


def _legacy_equations() -> dict[str, np.ndarray]:
    """Every legacy equation function evaluated on a grid around its working range."""
    q = _grid(0.0, 100.0)
    v = _grid(-15.0, 5.0)
    xs = _grid(-1.2, 1.2)
    ys = _grid(-1.2, 1.2)[::-1]
    h = _grid(10.0, 16.0)
    e = np.concatenate([[0.0], np.geomspace(1e-4, 10.0, 40)])
    lux = np.concatenate([[0.0], np.geomspace(1.0, 1e4, 40)])
    p = _grid(0.0, 1.0)
    alpha = _grid(0.0, 1e-3)
    theta_l = _grid(constants.NU_LA, 0.0)
    phi = _grid(-np.pi, np.pi, 73)
    t = _grid(0.0, 2 * DAY, 289)
    rho = 25.0 + 20.0 * np.sin(t / DAY * 2 * np.pi)

    # X == 0 would divide by zero in circadian_phase; the grid avoids it.
    phase_x = np.where(np.abs(xs) < 1e-9, 1e-3, xs)

    return {
        "wake_effort_forced": _each(models.wake_effort, q, 1),
        "wake_effort_free": _each(models.wake_effort, q, 0),
        "total_sleep_drive": _each(models.total_sleep_drive, h, _grid(0.0, 2.0)),
        "nonphotic_drive_wake": _each(models.nonphotic_drive, xs, 1),
        "nonphotic_drive_sleep": _each(models.nonphotic_drive, xs, 0),
        "photoreceptor_2020_wake": _each(models.photoreceptor_conversion_rate, e, 1, "2020"),
        "photoreceptor_2020_sleep": _each(models.photoreceptor_conversion_rate, e, 0, "2020"),
        "photoreceptor_2018_wake": _each(models.photoreceptor_conversion_rate, lux, 1, "2018"),
        "photic_drive": _each(models.photic_drive, xs, ys, p, alpha),
        "firing_rate": _each(models.mean_population_firing_rate, v),
        "state": _each(models.state, v),
        "sigmoid": _each(models.sigmoid, e),
        "alertness": _each(models.alertness_measure, _grid(0.0, 2.0), h, theta_l),
        "circadian_drive": _each(models.circadian_drive, xs, ys),
        "melatonin_suppression": _each(models.melatonin_suppression, e),
        "circadian_phase": _each(models.circadian_phase, phase_x, ys),
        "melatonin_synthesis": _each(models.melatonin_synthesis_regulation, phi),
        "forced_wake_6_22": _each(models.forced_wake, t, 6, 22),
        "forced_wake_8_20": _each(models.forced_wake, t, 8, 20),
        "urinary_excretion": np.array([models.urinary_excretion_rate(rho, t, ti) for ti in t]),
    }


def test_equations(ndarrays_regression: NDArraysRegressionFixture) -> None:
    ndarrays_regression.check(
        _legacy_equations(),
        basename="equations",
        default_tolerance=EQUATION_TOLERANCE,
    )
