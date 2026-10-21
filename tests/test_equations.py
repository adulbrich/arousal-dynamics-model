"""Each equation against values worked out by hand from the paper's formula.

The expected values use the papers' constants typed out here, not the
package's parameter object, so a wrong default in `ArousalParameters` fails a
test instead of agreeing with itself.
"""

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from arousal_dynamics import ArousalModel, ArousalParameters, Inputs, LightInput, WakeSchedule
from arousal_dynamics.models import arousal as eq

P = ArousalParameters()


class TestFiringRate:
    """Q = Q_max / (1 + exp((theta - V) / sigma')), P18 Table 1 Eq 7."""

    def test_half_maximum_at_threshold(self) -> None:
        assert eq.firing_rate(10.0, P) == pytest.approx(50.0)

    def test_three_quarters_one_log3_above(self) -> None:
        assert eq.firing_rate(10.0 + 3.0 * math.log(3), P) == pytest.approx(75.0)

    def test_quiet_far_below_threshold(self) -> None:
        assert eq.firing_rate(-20.0, P) == pytest.approx(100.0 / (1 + math.exp(10.0)))


class TestDrives:
    def test_total_sleep_drive(self) -> None:
        # nu_vh H + nu_vc C + A_v, P18 Table 1 Eq 9
        assert eq.total_sleep_drive(1.0, 1.0, P) == pytest.approx(1.0 - 0.5 - 10.3)
        assert eq.total_sleep_drive(25.0, 2.5, P) == pytest.approx(25.0 - 1.25 - 10.3)
        assert eq.total_sleep_drive(0.0, 0.0, P) == pytest.approx(-10.3)

    @pytest.mark.parametrize(
        ("X", "S", "expected"),
        [
            (0.0, 1, 1 / 3),
            (0.0, 0, -2 / 3),
            (-0.14, 1, (1 / 3) * (1 - math.tanh(-1.4))),
            (0.5, 0, (-2 / 3) * (1 - math.tanh(5.0))),
        ],
    )
    def test_nonphotic_drive(self, X: float, S: int, expected: float) -> None:
        # (S - 2/3)(1 - tanh(r X)), r = 10, P18 Table 1 Eq 11
        assert eq.nonphotic_drive(X, S, P) == pytest.approx(expected)

    def test_photic_drive(self) -> None:
        # alpha (1 - P)(1 - eps X)(1 - eps Y), eps = 0.4, P18 Table 1 Eq 12
        expected = 0.00016 * 0.9 * (1 + 0.4 * 0.14) * (1 + 0.4 * 1.07)
        assert eq.photic_drive(-0.14, -1.07, 0.10, 0.00016, P) == pytest.approx(expected)

    def test_circadian_drive_at_origin(self) -> None:
        # 0.1 (1 + X)/2 + ((3.1 X - 2.5 Y + 4.2) / (3.7 (X + 2)))^2, P18 Table 1 Eq 10
        assert eq.circadian_drive(0.0, 0.0) == pytest.approx(0.05 + (4.2 / 7.4) ** 2)


class TestWakeEffort:
    """W = F_w max(0, V_WE - nu_mv Q_v - D_m), P18 Table 1 Eq 8."""

    def test_forced(self) -> None:
        assert eq.wake_effort(30.0, 1, P) == pytest.approx(-0.07 + 1.8 * 30 - 1.3)

    def test_not_forced_is_zero(self) -> None:
        assert eq.wake_effort(30.0, 0, P) == 0.0

    def test_never_negative(self) -> None:
        assert eq.wake_effort(0.0, 1, P) == 0.0


class TestPhotoreceptor:
    def test_2018_photopic(self) -> None:
        # alpha_0 (I / 9500)^(1/2) I / (I + 100), SH07 Eq 9 (issue #6)
        p = replace(P, photic_version="2018")
        alpha_0 = 0.1 / 60
        for lux in (1.0, 100.0, 1000.0):
            expected = alpha_0 * math.sqrt(lux / 9500) * lux / (lux + 100)
            assert eq.photoreceptor_conversion_rate(lux, 1, p) == pytest.approx(expected)

    def test_2020_melanopic(self) -> None:
        # The 2018 form with I_0 and I_1 scaled by F_4100K, T20 Eq 7
        e, f = 0.1, 8.19e-4
        expected = (0.1 / 60) * math.sqrt(e / (9500 * f)) * e / (e + 100 * f)
        assert eq.photoreceptor_conversion_rate(e, 1, P) == pytest.approx(expected)

    def test_grows_as_i_to_the_1_5_in_dim_light(self) -> None:
        # SH07, text after Eq 9: alpha grows as I^1.5 well below 100 lux.
        p = replace(P, photic_version="2018")
        ratio = eq.photoreceptor_conversion_rate(0.02, 1, p) / eq.photoreceptor_conversion_rate(
            0.01, 1, p
        )
        assert ratio == pytest.approx(2**1.5, rel=1e-3)

    @pytest.mark.parametrize("version", ["2018", "2020"])
    def test_asleep_sees_no_light(self, version: str) -> None:
        p = replace(P, photic_version=version)
        assert eq.photoreceptor_conversion_rate(500.0, 0, p) == 0.0

    def test_unknown_version_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="photic_version"):
            eq.photoreceptor_conversion_rate(1.0, 1, replace(P, photic_version="2019"))


class TestSleepState:
    def test_threshold_is_strict(self) -> None:
        # S = 1 when V_m > V_th = -2 mV, P18 Table 1 Eq 15
        assert eq.sleep_state(-2.0, P) == 0
        assert eq.sleep_state(-1.999, P) == 1
        assert eq.sleep_state(-10.0, P) == 0


class TestAlertness:
    @staticmethod
    def _normalized(logistic: float) -> float:
        # T20 p. 7: the logistic less its value in darkness, rescaled to [0, 1]
        dark = 1 / (1 + math.exp(0.05 * 223.5))
        return (logistic - dark) / (1 - dark)

    def test_sigmoid_at_s_b(self) -> None:
        # 1 / (1 + exp((S_b - E) / S_c)), T20 Eq 14, is 1/2 at E = S_b before normalizing
        assert eq.alertness_sigmoid(0.05, P) == pytest.approx(self._normalized(0.5))

    def test_sigmoid_one_s_c_above(self) -> None:
        expected = self._normalized(1 / (1 + math.exp(-1.0)))
        assert eq.alertness_sigmoid(0.05 + 1 / 223.5, P) == pytest.approx(expected)

    def test_kss(self) -> None:
        # Theta_0 + (Theta_H + Theta_L) H + Theta_C C, T20 Eq 12
        expected = -24.34 + (2.28 - 0.11) * 13.0 + (-1.74) * 0.5
        assert eq.kss(0.5, 13.0, -0.11, P) == pytest.approx(expected)


class TestMelatonin:
    def test_synthesis_window(self) -> None:
        # m = 0 inside [phi_on, phi_off] = [-1.44, 2.78], 1 outside, A18 Eq 2
        assert eq.melatonin_synthesis(0.0, P) == 0
        assert eq.melatonin_synthesis(-1.44, P) == 0
        assert eq.melatonin_synthesis(2.78, P) == 0
        assert eq.melatonin_synthesis(-1.5, P) == 1
        assert eq.melatonin_synthesis(3.0, P) == 1

    def test_suppression_is_half_at_r_b(self) -> None:
        # The code's logistic form is 0.5 at E = R_b (issue #9 for the form).
        assert eq.melatonin_suppression(0.031, P) == pytest.approx(0.5)

    def test_excretion_of_a_steady_level(self) -> None:
        # u = u* rho_b(t - T_U) / rho_b*, A18 Eq 15
        t = np.linspace(0.0, 10 * 3600.0, 101)
        u = eq.urinary_excretion(np.full_like(t, 162.5), t, P)
        np.testing.assert_allclose(u, 0.47 * 162.5 / 325)

    def test_excretion_lags_by_t_u(self) -> None:
        t = np.arange(0.0, 6 * 3600.0, 60.0)
        rho = np.where(t >= 3600.0, 100.0, 0.0)
        u = eq.urinary_excretion(rho, t, P)
        lag = 0.96 * 3600
        assert np.all(u[t < 3600.0 + lag] == 0.0)
        assert np.all(u[t >= 3600.0 + lag + 60.0] == pytest.approx(0.47 * 100 / 325))

    def test_excretion_before_the_trajectory_uses_its_first_sample(self) -> None:
        t = np.array([0.0, 60.0, 120.0])
        u = eq.urinary_excretion(np.array([10.0, 20.0, 30.0]), t, P)
        np.testing.assert_allclose(u, 0.47 * 10.0 / 325)


class TestForcedWake:
    def test_bounds_are_inclusive(self) -> None:
        s = WakeSchedule(waketime=6, bedtime=22)
        hours = np.array([5.99, 6.0, 12.0, 22.0, 22.01])
        np.testing.assert_array_equal(s.forced_wake(hours * 3600), [0, 1, 1, 1, 0])

    def test_repeats_daily(self) -> None:
        s = WakeSchedule(waketime=8, bedtime=20)
        t = np.arange(0.0, 86400.0, 900.0)
        np.testing.assert_array_equal(s.forced_wake(t), s.forced_wake(t + 3 * 86400.0))

    @pytest.mark.parametrize(("wake", "bed"), [(22, 6), (8, 8), (-1, 10), (6, 25)])
    def test_unsupported_windows_are_rejected(self, wake: float, bed: float) -> None:
        with pytest.raises(ValueError, match="waketime"):
            WakeSchedule(waketime=wake, bedtime=bed)

    def test_no_schedule_never_forces_wake(self) -> None:
        # Asleep: VLPO active, MA quiet. A window would push V_m up; no window leaves W = 0.
        y = np.array([2.0, -11.7, 13.0, -0.6, -1.0, 0.25, 0.0, 0.0, 10.0])
        model, light, noon = ArousalModel(), LightInput.constant(0.0), 12 * 3600.0
        free = model.derivatives(noon, y, P, Inputs(light, schedule=None))
        forced = model.derivatives(noon, y, P, Inputs(light, WakeSchedule(0, 24)))
        unforced = (P.nu_mv * eq.firing_rate(2.0, P) + 11.7 + P.d_m) / P.tau_m
        assert free[1] == pytest.approx(unforced)
        assert forced[1] > free[1]


SCALAR_AND_ARRAY_CASES = [
    ("firing_rate", lambda v: eq.firing_rate(v, P), np.linspace(-15, 5, 9)),
    ("sleep_state", lambda v: eq.sleep_state(v, P), np.linspace(-5, 1, 9)),
    ("wake_effort", lambda q: eq.wake_effort(q, 1, P), np.linspace(0, 100, 9)),
    ("alertness_sigmoid", lambda e: eq.alertness_sigmoid(e, P), np.geomspace(1e-3, 1, 9)),
    ("melatonin_suppression", lambda e: eq.melatonin_suppression(e, P), np.geomspace(1e-3, 1, 9)),
    ("melatonin_synthesis", lambda phi: eq.melatonin_synthesis(phi, P), np.linspace(-3, 3, 9)),
    (
        "photoreceptor",
        lambda e: eq.photoreceptor_conversion_rate(e, 1, P),
        np.geomspace(1e-3, 1, 9),
    ),
    ("nonphotic_drive", lambda x: eq.nonphotic_drive(x, 1, P), np.linspace(-1, 1, 9)),
    ("circadian_drive", lambda x: eq.circadian_drive(x, -x), np.linspace(-1, 1, 9)),
    ("circadian_phase", lambda x: eq.circadian_phase(x, 0.5), np.linspace(0.1, 1, 9)),
]


@pytest.mark.parametrize(
    ("f", "grid"),
    [(f, g) for _, f, g in SCALAR_AND_ARRAY_CASES],
    ids=[name for name, _, _ in SCALAR_AND_ARRAY_CASES],
)
def test_scalar_and_array_agree(f: Callable[[Any], Any], grid: np.ndarray) -> None:
    """The ODE calls each equation with scalars; outputs use arrays. Both must agree exactly."""
    vectorized = np.asarray(f(grid), dtype=float)
    pointwise = np.array([float(f(float(x))) for x in grid])
    np.testing.assert_array_equal(vectorized, pointwise)
