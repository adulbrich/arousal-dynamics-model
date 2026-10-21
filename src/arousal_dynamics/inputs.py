"""What drives a simulation: the light a person sees and when they are kept awake."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

from arousal_dynamics.units import SECONDS_PER_HOUR, FloatArray, LightUnit, to_irradiance

type Interpolation = Literal["previous", "linear", "cubic"]

#: Time in seconds, a float or an array when vectorized, to light.
type LightFunction = Callable[[Any], Any]


@dataclass(frozen=True)
class LightInput:
    """Melanopic irradiance in W/m^2 as a function of time in seconds.

    Build one with `from_samples`, `constant` or `from_function` rather than
    directly, so that the unit conversion happens once, at the boundary.
    Negative values, which a cubic interpolant produces around steps, are
    clipped to zero when the light is read.

    Parameters
    ----------
    function
        Maps time in seconds to melanopic irradiance in W/m^2.
    vectorized
        Whether `function` accepts an array of times. A scalar-only function is
        evaluated point by point by `values`.
    """

    function: LightFunction
    vectorized: bool = False

    def __call__(self, t: float) -> float:
        """Light at one time point, clipped at zero."""
        return max(self.function(t), 0)

    def values(self, t: npt.ArrayLike) -> FloatArray:
        """Light at each of `t`, clipped at zero."""
        t = np.asarray(t, dtype=float)
        if self.vectorized:
            raw = np.asarray(self.function(t), dtype=float)
        else:
            raw = np.array([self.function(ti) for ti in t], dtype=float)
        return np.maximum(raw, 0.0)

    @classmethod
    def from_samples(
        cls,
        t: npt.ArrayLike,
        values: npt.ArrayLike,
        *,
        unit: LightUnit = LightUnit.MELANOPIC_IRRADIANCE,
        interpolation: Interpolation = "cubic",
    ) -> "LightInput":
        """Interpolate a sampled light series.

        Parameters
        ----------
        t
            Sample times in seconds, increasing.
        values
            Light at each sample, in `unit`.
        unit
            The unit of `values`; converted to W/m^2 here.
        interpolation
            How to read between samples. `"cubic"` reproduces the original
            `model_run`; `"previous"` holds each sample until the next one,
            which suits a step schedule. Outside the samples the first and last
            values are held.
        """
        t = np.asarray(t, dtype=float)
        e = to_irradiance(np.asarray(values, dtype=float), unit)
        if t.ndim != 1 or t.shape != e.shape:
            msg = f"t and values must be 1-D and the same length, got {t.shape} and {e.shape}"
            raise ValueError(msg)
        minimum = {"previous": 2, "linear": 2, "cubic": 4}[interpolation]
        if t.size < minimum:
            msg = f"{interpolation!r} interpolation needs at least {minimum} samples, got {t.size}"
            raise ValueError(msg)
        interpolator = interp1d(
            t,
            e,
            kind=interpolation,
            bounds_error=False,
            fill_value=(e[0], e[-1]),
        )
        return cls(interpolator, vectorized=True)

    @classmethod
    def constant(
        cls, value: float, *, unit: LightUnit = LightUnit.MELANOPIC_IRRADIANCE
    ) -> "LightInput":
        """Build light that is the same at every time."""
        e = float(to_irradiance(value, unit))
        return cls(lambda t: np.full_like(np.asarray(t, dtype=float), e), vectorized=True)

    @classmethod
    def from_function(
        cls,
        function: LightFunction,
        *,
        unit: LightUnit = LightUnit.MELANOPIC_IRRADIANCE,
        vectorized: bool = False,
    ) -> "LightInput":
        """Wrap a function of time in seconds that returns light in `unit`."""
        if unit is LightUnit.MELANOPIC_IRRADIANCE:
            return cls(function, vectorized=vectorized)
        return cls(lambda t: to_irradiance(function(t), unit), vectorized=vectorized)


@dataclass(frozen=True)
class WakeSchedule:
    """A daily window, in clock hours, during which wake is enforced.

    Postnova et al. 2018 apply a wake effort while the forced-wake signal is on,
    which keeps the model awake through its scheduled day even when sleep
    pressure would flip it. Both bounds are inclusive.

    Parameters
    ----------
    waketime
        Clock hour the forced-wake window opens, in [0, 24).
    bedtime
        Clock hour it closes, in (waketime, 24]. A window across midnight is
        not supported yet.
    """

    waketime: float = 6.0
    bedtime: float = 22.0

    def __post_init__(self) -> None:
        """Reject windows the forced-wake rule cannot express."""
        if not 0.0 <= self.waketime < self.bedtime <= 24.0:
            msg = (
                f"need 0 <= waketime < bedtime <= 24, got waketime={self.waketime}, "
                f"bedtime={self.bedtime}; a window across midnight is not supported yet"
            )
            raise ValueError(msg)

    def forced_wake(self, t: npt.ArrayLike) -> npt.NDArray[np.int_]:
        """1 while wake is enforced at time `t` in seconds, else 0.

        Table 1 in Postnova et al. 2018.
        """
        clock = np.asarray(t, dtype=float) / SECONDS_PER_HOUR % 24
        return np.where((clock >= self.waketime) & (clock <= self.bedtime), 1, 0)


@dataclass(frozen=True)
class Inputs:
    """Everything outside the model that a simulation reads.

    `schedule=None` enforces no wake at all: the model falls asleep and wakes
    on its own, as in the entrainment protocol of Postnova et al. 2016 (p. 500)
    and 2018 (p. 204).
    """

    light: LightInput
    schedule: WakeSchedule | None = field(default_factory=WakeSchedule)
