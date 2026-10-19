"""Light-driven arousal dynamics: sleep/wake, circadian phase, melatonin and sleepiness.

>>> from arousal_dynamics import ArousalModel, Inputs, LightInput, LightUnit, simulate, time_grid
>>> light = LightInput.constant(100, unit=LightUnit.MELANOPIC_EDI)
>>> run = simulate(ArousalModel(), Inputs(light), time_grid(days=1, steps=97))
>>> run["KSS"].shape
(97,)
"""

from arousal_dynamics.inputs import Inputs, LightInput, WakeSchedule
from arousal_dynamics.models.arousal import ArousalModel, ArousalParameters, time_grid
from arousal_dynamics.simulation import IntegrationError, Model, Trajectory, simulate
from arousal_dynamics.units import LightUnit, irradiance_to_mel_edi, mel_edi_to_irradiance

__all__ = [
    "ArousalModel",
    "ArousalParameters",
    "Inputs",
    "IntegrationError",
    "LightInput",
    "LightUnit",
    "Model",
    "Trajectory",
    "WakeSchedule",
    "irradiance_to_mel_edi",
    "mel_edi_to_irradiance",
    "simulate",
    "time_grid",
]
