# Light-Driven Arousal Dynamics Model

A Python implementation of the arousal dynamics model: given the light a person sees and
when they are kept awake, it predicts sleep and wake, circadian phase, plasma melatonin,
urinary 6-sulfatoxymelatonin (aMT6s), and subjective sleepiness on the Karolinska
Sleepiness Scale (KSS). Equations and parameters come from:

[Postnova et al. 2016] Postnova, S., Lockley, S. W., & Robinson, P. A. (2016). Sleep Propensity under Forced Desynchrony in a Model of Arousal State Dynamics. _Journal of Biological Rhythms_, _31_(5), 498–508. https://doi.org/10.1177/0748730416658806

[Abeysuriya et al. 2018] Abeysuriya RG, Lockley SW, Robinson PA, Postnova S. A unified model of melatonin, 6-sulfatoxymelatonin, and sleep dynamics. _Journal of Pineal Research_, _64_(4), e12474. https://doi.org/10.1111/jpi.12474

[Postnova et al. 2018] Postnova, S., Lockley, S. W., & Robinson, P. A. (2018). Prediction of Cognitive Performance and Subjective Sleepiness Using a Model of Arousal Dynamics. _Journal of Biological Rhythms_, _33_(2), 203–218. https://doi.org/10.1177/0748730418758454

[St Hilaire et al. 2007] St Hilaire, M. A., Klerman, E. B., Khalsa, S. B., Wright, K. P., Czeisler, C. A., & Kronauer, R. E. (2007). Addition of a non-photic component to a light-based mathematical model of the human circadian pacemaker. _Journal of Theoretical Biology_, _247_(4), 583–599. https://doi.org/10.1016/j.jtbi.2007.04.001

[Tekieh et al. 2020] Tekieh, T., Lockley, S. W., Robinson, P. A., McCloskey, S., Zobaer, M. S., & Postnova, S. (2020). Modeling melanopsin‐mediated effects of light on circadian phase, melatonin suppression, and subjective sleepiness. _Journal of Pineal Research_, _69_(3), e12681. https://doi.org/10.1111/jpi.12681

Project led by Clotilde Pierson. Implementation by Alexander Ulbrich.

## Install

The package is not on PyPI. Add it to a uv project straight from GitHub, with the
optional plotting and pandas support:

```sh
uv add "arousal-dynamics[plot,pandas] @ git+https://github.com/adulbrich/arousal-dynamics-model"
```

It needs Python 3.12 or newer. The core depends on NumPy and SciPy only.

## Quickstart

```python
import numpy as np
from arousal_dynamics import (
    ArousalModel,
    Inputs,
    LightInput,
    LightUnit,
    WakeSchedule,
    simulate,
)

hours = np.arange(0.0, 8 * 24.0, 0.25)
clock = hours % 24
lux = np.where((clock >= 8) & (clock < 18), 250.0, 5.0)  # melanopic EDI
light = LightInput.from_samples(
    hours * 3600, lux, unit=LightUnit.MELANOPIC_EDI, interpolation="previous"
)

inputs = Inputs(light, WakeSchedule(waketime=7, bedtime=23))
t = np.arange(0.0, 7 * 86400 + 1, 600.0)  # seconds

run = simulate(ArousalModel(), inputs, t)
run["KSS"], run["rho_b"], run["S"]  # any state or output, by name
run.to_dataframe()  # with the pandas extra
```

Parameters are a frozen dataclass holding the papers' values, each with its unit and
source in the field metadata. Override any of them per run:

```python
from dataclasses import replace
from arousal_dynamics import ArousalParameters

variant = simulate(ArousalModel(), inputs, t, params=replace(ArousalParameters(), s_b=0.3))
```

`notebooks/quickstart.ipynb` walks through the same steps with plots, and CI runs it.

## What is in the package

| Module | Contents |
| --- | --- |
| `arousal_dynamics.models.arousal` | `ArousalModel`, `ArousalParameters`, and every equation as a function citing its paper and equation number |
| `arousal_dynamics.simulation` | `simulate`, `Trajectory`, and the `Model` interface a second model would implement |
| `arousal_dynamics.inputs` | `LightInput` (light with units and interpolation), `WakeSchedule`, `Inputs` |
| `arousal_dynamics.units` | Melanopic EDI and melanopic irradiance conversion (CIE S 026) |
| `arousal_dynamics.plotting` | `plot_light`, `plot_trajectory` (the `plot` extra) |

`docs/adr/0001-one-model-behind-a-small-interface.md` explains the shape and what is
left out on purpose.

## Scope

The model was fitted mostly to evening and night-time light and does not tell an office
day from an outdoor day;
`docs/adr/0002-keep-the-arousal-model-to-the-light-it-was-fitted-on.md` records why it
stays that way for now.

## Known discrepancies with the papers

The code reproduces the original implementation exactly, including where it departs from
the papers. The equation audit ([issue #7][audit]) checks every equation and constant
against the four papers; each discrepancy is an issue of its own, with a recommendation.
The two code bugs, the circadian phase ([#8][i8]) and the melatonin suppression form
([#9][i9]), are tests marked as expected failures in `tests/test_invariants.py`; each fix
lands in its own pull request.

[audit]: https://github.com/adulbrich/arousal-dynamics-model/issues/7
[i8]: https://github.com/adulbrich/arousal-dynamics-model/issues/8
[i9]: https://github.com/adulbrich/arousal-dynamics-model/issues/9

Two numerical facts worth knowing before comparing against data. From its default initial
state the model takes about two weeks to entrain to a regular schedule, so start a
comparison after a run-in. And the solver is `odeint` (LSODA) as in the original code;
the sleep/wake switch and the forced-wake window are discontinuities an event-aware solver
would handle better, which will come as a numerical change with its own tolerance study.

## Development

The project is managed with [uv](https://docs.astral.sh/uv/) 0.13.2 or newer:

```sh
uv sync
uv run lefthook install
uv run pytest
```

`CONTRIBUTING.md` has the process, the golden-output rule, the test tiers and the gates.
For the notebooks, `uv sync --group notebooks` and `uv run jupyter lab`;
`notebooks/archive/` holds the analyses from before the package, which no longer run.

## Open work

Open work is tracked in the [issues][issues], not in this README: the audited discrepancies
with the papers ([issue #7][audit]), the model landscape and a future daytime model, and
engineering work such as an entrainment helper and an event-aware solver. The pull request
that finishes an item closes its issue.

[issues]: https://github.com/adulbrich/arousal-dynamics-model/issues
