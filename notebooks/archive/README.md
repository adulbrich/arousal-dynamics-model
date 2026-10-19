# Archived notebooks

These notebooks predate the `arousal_dynamics` package. They import the flat
`models.py` and `constants.py` modules, which were replaced by the package in the
library refactor, so they no longer run as they are. They are kept as the record
of earlier analyses, outputs included.

To run one as it was, check out the last commit before the refactor and use the
modules it shipped with:

```sh
git switch --detach 55eeb3e
```

`validation.ipynb` compared the model's KSS with `data/output.csv`;
`tests/test_validation.py` now makes that comparison numerically.
`sigmoid_fit.ipynb` and `sigmoid.py` refit the Tekieh 2020 alertness sigmoid
(see the equation audit, E3); the package takes the refit values as a parameter
override, `dataclasses.replace(ArousalParameters(), s_b=..., s_c=...)`, so the
copy of the model in `sigmoid.py` is no longer needed.
