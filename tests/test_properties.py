"""Invariants every equation must hold for any input in its physiological range."""

import math
from dataclasses import replace

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

from arousal_dynamics import (
    ArousalParameters,
    LightInput,
    LightUnit,
    WakeSchedule,
    irradiance_to_mel_edi,
    mel_edi_to_irradiance,
)
from arousal_dynamics.models import arousal as eq

P = ArousalParameters()

voltage = st.floats(-30.0, 30.0)
irradiance = st.floats(0.0, 100.0)  # W/m^2; bright daylight is about 1
oscillator = st.floats(-1.5, 1.5)
occupancy = st.floats(0.0, 1.0)
awake = st.sampled_from([0, 1])
seconds = st.floats(0.0, 365 * 86400.0)


@given(voltage)
def test_firing_rate_is_bounded(V: float) -> None:
    assert 0.0 <= eq.firing_rate(V, P) <= P.q_max


@given(voltage, voltage)
def test_firing_rate_increases_with_voltage(a: float, b: float) -> None:
    lo, hi = sorted((a, b))
    assert eq.firing_rate(lo, P) <= eq.firing_rate(hi, P)


@given(voltage)
def test_sleep_state_is_binary_and_thresholded(V: float) -> None:
    assert int(eq.sleep_state(V, P)) == int(P.v_th < V)


@given(irradiance)
def test_alertness_sigmoid_is_a_fraction(E: float) -> None:
    assert 0.0 <= eq.alertness_sigmoid(E, P) <= 1.0


@given(irradiance, irradiance)
def test_alertness_sigmoid_increases_with_light(a: float, b: float) -> None:
    lo, hi = sorted((a, b))
    assert eq.alertness_sigmoid(lo, P) <= eq.alertness_sigmoid(hi, P)


@given(irradiance)
def test_melatonin_suppression_is_a_fraction(E: float) -> None:
    assert 0.0 <= eq.melatonin_suppression(E, P) <= 1.0


@given(irradiance, irradiance)
def test_more_light_suppresses_more(a: float, b: float) -> None:
    lo, hi = sorted((a, b))
    assert eq.melatonin_suppression(hi, P) <= eq.melatonin_suppression(lo, P)


@given(irradiance, awake, st.sampled_from(["2018", "2020"]))
def test_photoreceptor_rate_is_nonnegative_and_zero_asleep(E: float, S: int, version: str) -> None:
    p = replace(P, photic_version=version)
    alpha = eq.photoreceptor_conversion_rate(E, S, p)
    assert alpha >= 0.0
    if S == 0:
        assert alpha == 0.0


@given(irradiance, irradiance)
def test_photoreceptor_rate_increases_with_light(a: float, b: float) -> None:
    lo, hi = sorted((a, b))
    assert eq.photoreceptor_conversion_rate(lo, 1, P) <= eq.photoreceptor_conversion_rate(hi, 1, P)


@given(oscillator, oscillator, occupancy, st.floats(0.0, 1e-2))
def test_photic_drive_is_nonnegative_in_the_oscillator_range(
    X: float, Y: float, occ: float, alpha: float
) -> None:
    # (1 - eps X)(1 - eps Y) > 0 while |X|, |Y| < 1 / eps = 2.5
    assert eq.photic_drive(X, Y, occ, alpha, P) >= 0.0


@given(oscillator, awake)
def test_nonphotic_drive_bounds(X: float, S: int) -> None:
    # (S - 2/3)(1 - tanh(r X)) with 1 - tanh in (0, 2)
    d = eq.nonphotic_drive(X, S, P)
    if S == 1:
        assert 0.0 <= d <= 2 / 3
    else:
        assert -4 / 3 <= d <= 0.0


@given(st.floats(0.0, 200.0), awake)
def test_wake_effort_is_nonnegative_and_only_when_forced(Q_v: float, forced: int) -> None:
    w = eq.wake_effort(Q_v, forced, P)
    assert w >= 0.0
    if forced == 0:
        assert w == 0.0


@given(oscillator, oscillator)
def test_circadian_phase_stays_in_the_two_quadrant_range(X: float, Y: float) -> None:
    # Documents the current `arctan(Y / X)`; audit D1 says the papers need
    # the full circle, and the fix replaces this test.
    assume(abs(X) > 1e-6)
    phi = eq.circadian_phase(X, Y)
    assert -math.pi / 2 <= phi <= math.pi / 2


@given(seconds, st.integers(-30, 30))
def test_forced_wake_repeats_every_day(t: float, days: int) -> None:
    assume(t + days * 86400.0 >= 0)
    s = WakeSchedule(waketime=7.5, bedtime=23.25)
    # Away from the window edges, where float rounding of t mod 24 h could flip it.
    clock = t / 3600 % 24
    assume(min(abs(clock - 7.5), abs(clock - 23.25)) > 1e-6)
    assert s.forced_wake(t) == s.forced_wake(t + days * 86400.0)


# Zero, or a microlux to bright daylight: below that the division underflows
# into subnormal floats, which no light meter reports.
@given(st.just(0.0) | st.floats(1e-6, 1e5))
def test_mel_edi_round_trip(lux: float) -> None:
    assert math.isclose(irradiance_to_mel_edi(mel_edi_to_irradiance(lux)), lux, rel_tol=1e-15)


@given(st.lists(st.floats(-1.0, 10.0), min_size=4, max_size=30), st.floats(0.0, 1.0))
def test_sampled_light_is_never_negative(values: list[float], where: float) -> None:
    t = np.arange(len(values)) * 600.0
    light = LightInput.from_samples(t, values, unit=LightUnit.MELANOPIC_IRRADIANCE)
    probe = where * t[-1]
    assert light(probe) >= 0.0
    assert np.all(light.values(np.linspace(0, t[-1], 50)) >= 0.0)


@given(st.floats(0.0, 1e4))
def test_constant_light_converts_units_once(lux: float) -> None:
    light = LightInput.constant(lux, unit=LightUnit.MELANOPIC_EDI)
    assert math.isclose(float(light(1234.0)), mel_edi_to_irradiance(lux), rel_tol=1e-15)
