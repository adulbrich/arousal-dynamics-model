# Light-Driven Arousal Dynamics Model

This repository aims at implementing the arousal dynamics model as described in:

[Postnova et al. 2016] Postnova, S., Lockley, S. W., & Robinson, P. A. (2016). Sleep Propensity under Forced Desynchrony in a Model of Arousal State Dynamics. _Journal of Biological Rhythms_, _31_(5), 498–508. https://doi.org/10.1177/0748730416658806

[Abeysuriya et al. 2018] Abeysuriya RG, Lockley SW, Robinson PA, Postnova S. A unified model of melatonin, 6-sulfatoxymelatonin, and sleep dynamics. _Journal of Pineal Research_, _64_(4), e12474. https://doi.org/10.1111/jpi.12474 

[Postnova et al. 2018] Postnova, S., Lockley, S. W., & Robinson, P. A. (2018). Prediction of Cognitive Performance and Subjective Sleepiness Using a Model of Arousal Dynamics. _Journal of Biological Rhythms_, _33_(2), 203–218. https://doi.org/10.1177/0748730418758454

[Tekieh et al. 2020] Tekieh, T., Lockley, S. W., Robinson, P. A., McCloskey, S., Zobaer, M. S., & Postnova, S. (2020). Modeling melanopsin‐mediated effects of light on circadian phase, melatonin suppression, and subjective sleepiness. _Journal of Pineal Research_, _69_(3), e12681. https://doi.org/10.1111/jpi.12681

Parameters and equations are taken from the papers.

Project led by Clotilde Pierson. Implementation by Alexander Ulbrich.

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/). Install uv, then from the
repository root:

```sh
uv sync
```

This creates `.venv` with the Python version in `.python-version` and the locked
dependencies. Run anything inside it with `uv run`, for example `uv run pytest`. For the
notebooks:

```sh
uv sync --group notebooks
uv run jupyter lab
```

Contributing, the git hooks and the CI gates are described in `CONTRIBUTING.md`.

## To Do

- [ ] optimize for S_b and S_c constants for daytime
- [ ] add debug flags where relevant
- [ ] clean up old notebooks (won't run on current models.py)
- [ ] add more unit tests, including comparison with original data if possible (golden tests pin current outputs)
- [ ] have a notebook to run on scenarios and compare outputs (aMT6s, rho_b, KSS)
- [ ] improve plots with better names for axes
- [ ] re-organize `constants` in particular for the melatonin concentrations
- [ ] add missing docstrings and fix linting in the legacy modules (ruff and ty run on new code)
