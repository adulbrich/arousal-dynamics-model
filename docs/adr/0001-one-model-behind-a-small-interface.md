# 0001. One model behind a small interface

The repository becomes a library that may host other light-response models later, but no
second model is chosen yet, so the package implements exactly one, `ArousalModel`, behind
a `Model` protocol in `arousal_dynamics.simulation` that holds only what the model
landscape (`docs/research/2026-10-19-circadian-model-landscape.md`) shows every candidate
needs: named states, a parameter object, an initial state, a right-hand side, and derived
outputs computed over the whole trajectory. Light is an input with a unit, converted once
at the boundary, because candidates disagree between photopic lux, melanopic EDI and
melanopic irradiance; each model declares its time unit, because they disagree between
seconds and hours. There is no registry, plugin mechanism or shared parameter base class,
and integration stays `odeint` with its defaults so the golden outputs pin the refactor.
The interface is proven only by a toy model in `tests/test_simulation.py`, so the second
real model will likely reshape it; that pull request, not this one, decides how, and the
solver change gets its own pull request with a tolerance study.
