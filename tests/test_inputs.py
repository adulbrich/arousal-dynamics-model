"""Building light inputs: units, interpolation, and the errors a caller can hit."""

import numpy as np
import pytest

from arousal_dynamics import LightInput, LightUnit, mel_edi_to_irradiance
from arousal_dynamics.units import MEL_EDI_LUX_PER_W_M2, to_irradiance


def test_lux_converts_through_the_d65_efficacy() -> None:
    # CIE S 026: 1.3262 mW/lm, so 1 W/m^2 is 754.03 lux of melanopic EDI.
    assert pytest.approx(1 / 1.3262e-3, rel=1e-8) == MEL_EDI_LUX_PER_W_M2
    assert to_irradiance(754.0340838, LightUnit.MELANOPIC_EDI) == pytest.approx(1.0)
    assert to_irradiance(0.5, LightUnit.MELANOPIC_IRRADIANCE) == 0.5


class TestFromSamples:
    t = np.array([0.0, 3600.0, 7200.0, 10800.0, 14400.0])

    def test_previous_holds_each_sample(self) -> None:
        light = LightInput.from_samples(self.t, [0.0, 1.0, 0.0, 1.0, 0.0], interpolation="previous")
        np.testing.assert_array_equal(light.values([0.0, 3599.0, 3600.0, 5000.0]), [0, 0, 1, 1])

    def test_linear_interpolates(self) -> None:
        light = LightInput.from_samples(self.t, [0.0, 1.0, 0.0, 1.0, 0.0], interpolation="linear")
        assert float(light(1800.0)) == pytest.approx(0.5)

    def test_holds_the_ends_outside_the_samples(self) -> None:
        light = LightInput.from_samples(self.t, [0.2, 1.0, 0.0, 1.0, 0.4], interpolation="linear")
        np.testing.assert_allclose(light.values([-100.0, 1e6]), [0.2, 0.4])

    def test_converts_lux(self) -> None:
        light = LightInput.from_samples(
            self.t, [250.0] * 5, unit=LightUnit.MELANOPIC_EDI, interpolation="previous"
        )
        assert float(light(100.0)) == pytest.approx(mel_edi_to_irradiance(250.0))

    def test_cubic_overshoot_is_clipped_at_zero(self) -> None:
        light = LightInput.from_samples(self.t, [0.0, 0.0, 1.0, 0.0, 0.0], interpolation="cubic")
        dense = np.linspace(0, 14400.0, 500)
        assert light.values(dense).min() == 0.0

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            LightInput.from_samples(self.t, [1.0, 2.0])

    def test_cubic_needs_four_samples(self) -> None:
        with pytest.raises(ValueError, match="at least 4"):
            LightInput.from_samples([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], interpolation="cubic")


def test_from_function_converts_lux() -> None:
    light = LightInput.from_function(lambda t: 100.0 + 0 * t, unit=LightUnit.MELANOPIC_EDI)
    assert float(light(5.0)) == pytest.approx(mel_edi_to_irradiance(100.0))


def test_scalar_function_is_evaluated_point_by_point() -> None:
    calls: list[float] = []

    def f(t: float) -> float:
        calls.append(t)
        return -1.0 if t < 10 else 2.0

    light = LightInput.from_function(f)
    np.testing.assert_array_equal(light.values([0.0, 20.0]), [0.0, 2.0])
    assert calls == [0.0, 20.0]
