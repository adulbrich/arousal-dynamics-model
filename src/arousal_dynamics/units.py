"""Light units and time conversions.

The arousal model takes melanopic irradiance in W/m^2 (Tekieh et al. 2020). Field
and lab data usually report melanopic equivalent daylight illuminance (melanopic
EDI) in lux. CIE S 026 relates the two through the melanopic efficacy of
luminous radiation of D65, 1.3262 mW/lm, so 1 W/m^2 of melanopic irradiance is
754.034 lux of melanopic EDI.

>>> round(mel_edi_to_irradiance(754.0340838), 12)
1.0
>>> round(irradiance_to_mel_edi(0.1), 6)
75.403408
"""

from enum import StrEnum

import numpy as np
import numpy.typing as npt

#: Melanopic EDI in lux per W/m^2 of melanopic irradiance, the inverse of the
#: CIE S 026 melanopic efficacy of D65 (1.3262 mW/lm).
MEL_EDI_LUX_PER_W_M2 = 754.0340838

SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 24.0 * SECONDS_PER_HOUR

type FloatArray = npt.NDArray[np.float64]


class LightUnit(StrEnum):
    """The quantity a light series is expressed in."""

    MELANOPIC_IRRADIANCE = "W/m^2"
    """Melanopic irradiance, W/m^2. What the arousal model integrates."""

    MELANOPIC_EDI = "lux"
    """Melanopic equivalent daylight illuminance, lux (CIE S 026)."""


def mel_edi_to_irradiance[T: (float, FloatArray)](lux: T) -> T:
    """Convert melanopic EDI in lux to melanopic irradiance in W/m^2."""
    return lux / MEL_EDI_LUX_PER_W_M2


def irradiance_to_mel_edi[T: (float, FloatArray)](w_m2: T) -> T:
    """Convert melanopic irradiance in W/m^2 to melanopic EDI in lux."""
    return w_m2 * MEL_EDI_LUX_PER_W_M2


def to_irradiance[T: (float, FloatArray)](values: T, unit: LightUnit) -> T:
    """Express `values`, given in `unit`, as melanopic irradiance in W/m^2."""
    if unit is LightUnit.MELANOPIC_IRRADIANCE:
        return values
    return mel_edi_to_irradiance(values)
