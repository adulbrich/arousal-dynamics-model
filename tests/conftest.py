"""Shared pytest setup: Hypothesis profiles and one reusable multi-day run."""

import os

import numpy as np
import pytest
from hypothesis import settings

from arousal_dynamics import (
    ArousalModel,
    ArousalParameters,
    Inputs,
    LightInput,
    LightUnit,
    Trajectory,
    WakeSchedule,
    simulate,
)

# `dev` keeps a local run quick; CI sets HYPOTHESIS_PROFILE=ci for more examples
# and a fixed seed, so a red run reproduces from its log.
settings.register_profile("dev", max_examples=100, deadline=None)
settings.register_profile("ci", max_examples=400, deadline=None, derandomize=True, print_blob=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

# Plotting tests and the notebook run must never open a window. Set before any
# test imports pyplot; nothing above imports matplotlib.
os.environ.setdefault("MPLBACKEND", "Agg")

HOUR = 3600.0
DAY = 24 * HOUR


@pytest.fixture(autouse=True)
def _no_inherited_git_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test's git commands on the repository the test names.

    The pre-push hook runs this suite, and from a linked worktree git exports
    GIT_DIR to it. Inherited, that sends the hook tests' `git init`, `commit`
    and `worktree add` to the repository being pushed instead of their
    temporary ones.
    """
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)


def _office_inputs(days: int) -> Inputs:
    """Dim night (5 lux), bright day (250 lux) 08:00 to 18:00, 30 lux evening to 23:00."""
    hours = np.arange(0.0, (days + 1) * 24.0 + 0.25, 0.25)
    clock = hours % 24.0
    lux = np.select(
        [(clock >= 8) & (clock < 18), (clock >= 7) & (clock < 23)],
        [250.0, 30.0],
        default=5.0,
    )
    light = LightInput.from_samples(
        hours * HOUR, lux, unit=LightUnit.MELANOPIC_EDI, interpolation="previous"
    )
    return Inputs(light, WakeSchedule(waketime=7, bedtime=23))


@pytest.fixture(scope="session")
def office_week() -> Trajectory:
    """Five days of a regular office schedule in melanopic EDI, 10-minute output.

    Dim night (5 lux), bright day (250 lux) between 08:00 and 18:00, evening
    (30 lux) until bed at 23:00; forced wake 07:00 to 23:00.
    """
    return simulate(ArousalModel(), _office_inputs(5), np.arange(0.0, 5 * DAY + 1, 600.0))


@pytest.fixture(scope="session")
def office_fortnight() -> Trajectory:
    """The `office_week` schedule for 14 days at 15-minute output, for entrainment."""
    return simulate(ArousalModel(), _office_inputs(14), np.arange(0.0, 14 * DAY + 1, 900.0))


def p18_entrainment_inputs() -> Inputs:
    """The entrainment protocol of Postnova et al. 2016 (p. 500) and 2018 (p. 204).

    250 lux from 08:00 to 20:00 and 40 lux otherwise, photopic lux of a 4100 K
    fluorescent lamp, converted to melanopic irradiance with F_4100K (Tekieh et
    al. 2020, Eq 7). The papers set 0 lux during sleep; the model already gates
    light by the sleep state, so the input need not. No forced wake: sleep comes
    and goes on its own.
    """
    f_4100k = ArousalParameters().f_4100k

    def irradiance(t: np.ndarray) -> np.ndarray:
        clock = np.asarray(t, dtype=float) / HOUR % 24
        return np.where((clock >= 8) & (clock < 20), 250.0, 40.0) * f_4100k

    return Inputs(LightInput.from_function(irradiance, vectorized=True), schedule=None)


@pytest.fixture(scope="session")
def p18_entrainment() -> Trajectory:
    """60 days on the P16 and P18 entrainment protocol from the default state, 5-minute output."""
    return simulate(ArousalModel(), p18_entrainment_inputs(), np.arange(0.0, 60 * DAY + 1, 300.0))
