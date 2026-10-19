"""Run a model over time: the `Model` interface, `simulate`, and its `Trajectory`."""

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from scipy.integrate import ODEintWarning, odeint

from arousal_dynamics.inputs import Inputs
from arousal_dynamics.units import SECONDS_PER_HOUR, FloatArray

if TYPE_CHECKING:
    import pandas as pd


@runtime_checkable
class Model[P](Protocol):
    """What `simulate` needs from a model.

    A model owns its state names, parameters, initial state, equations and
    derived outputs; `simulate` owns integration. `ArousalModel` is the one
    implementation today. The interface is kept to what a second model would
    need without guessing which one that is (docs/adr/0001).
    """

    @property
    def name(self) -> str:
        """A short name for messages."""
        ...

    @property
    def state_names(self) -> tuple[str, ...]:
        """State names, in the order of the state vector."""
        ...

    @property
    def time_unit(self) -> str:
        """The unit of `t` the model's equations expect, for example "s"."""
        ...

    def default_parameters(self) -> P:
        """Return the model's published parameter set."""
        ...

    def initial_state(self, params: P) -> FloatArray:
        """Return the state at t = 0 when the caller gives none."""
        ...

    def derivatives(self, t: float, y: FloatArray, params: P, inputs: Inputs) -> Sequence[Any]:
        """dy/dt at time `t`."""
        ...

    def outputs(
        self, t: FloatArray, states: FloatArray, params: P, inputs: Inputs
    ) -> Mapping[str, FloatArray]:
        """Series derived from a whole trajectory, keyed by name."""
        ...


class IntegrationError(RuntimeError):
    """The solver gave up before reaching the last time point."""


@dataclass(frozen=True)
class Trajectory[P]:
    """A simulation's result: time, every state, and the model's derived outputs.

    Index by name for any state or output: `trajectory["KSS"]`,
    `trajectory["V_m"]`.
    """

    model: Model[P]
    params: P
    inputs: Inputs
    t: FloatArray
    states: FloatArray
    outputs: Mapping[str, FloatArray] = field(repr=False)

    @property
    def state_names(self) -> tuple[str, ...]:
        """The model's state names, in column order of `states`."""
        return self.model.state_names

    @property
    def names(self) -> tuple[str, ...]:
        """Every name `trajectory[name]` accepts: states, then outputs."""
        return (*self.state_names, *self.outputs)

    @property
    def t_hours(self) -> FloatArray:
        """Time in hours."""
        return self.t / SECONDS_PER_HOUR

    def __getitem__(self, name: str) -> FloatArray:
        """Return a state or derived output by name."""
        if name in self.state_names:
            return self.states[:, self.state_names.index(name)]
        if name in self.outputs:
            return self.outputs[name]
        msg = f"{name!r} is not a state or output of {self.model.name}; have {self.names}"
        raise KeyError(msg)

    def to_dataframe(self) -> "pd.DataFrame":
        """Every state and output as columns, indexed by time in seconds.

        Needs pandas: `pip install arousal-dynamics[pandas]`.
        """
        try:
            import pandas as pd  # noqa: PLC0415
        except ImportError as error:
            msg = "to_dataframe needs pandas: install arousal-dynamics[pandas]"
            raise ImportError(msg) from error
        frame = pd.DataFrame({name: self[name] for name in self.names}, index=self.t)
        frame.index.name = "t"
        return frame


def simulate[P](
    model: Model[P],
    inputs: Inputs,
    t: npt.ArrayLike,
    *,
    params: P | None = None,
    y0: npt.ArrayLike | None = None,
) -> Trajectory[P]:
    """Integrate `model` driven by `inputs` and report it at the times `t`.

    Parameters
    ----------
    model
        The model, for example `ArousalModel()`.
    inputs
        Light and wake schedule.
    t
        Output times in the model's time unit, strictly increasing; the first
        is the initial time. `arousal_dynamics.time_grid(days, steps)` builds
        the grid the original `model_run` used.
    params
        The parameter set; the model's defaults when omitted. Vary one with
        `dataclasses.replace(model.default_parameters(), name=value)`.
    y0
        The initial state; `model.initial_state(params)` when omitted.

    Returns
    -------
    Trajectory
        States at each of `t` plus the model's derived outputs.

    Raises
    ------
    ValueError
        If `t` or `y0` has the wrong shape, or `t` is not strictly increasing.
    IntegrationError
        If the solver stops early.

    Notes
    -----
    Integration is LSODA through `scipy.integrate.odeint` with its default
    tolerances, as the original code ran it, so results match the golden
    outputs. The model has discontinuous switches (sleep state, forced wake,
    melatonin synthesis) that an event-aware solver would handle better; that
    is a numerical change and will come with its own tolerance analysis.
    """
    params = model.default_parameters() if params is None else params
    times = np.asarray(t, dtype=float)
    if times.ndim != 1 or times.size < 2:
        msg = f"t must be 1-D with at least two points, got shape {times.shape}"
        raise ValueError(msg)
    if np.any(np.diff(times) <= 0):
        msg = "t must be strictly increasing"
        raise ValueError(msg)

    start = model.initial_state(params) if y0 is None else np.asarray(y0, dtype=float)
    if start.shape != (len(model.state_names),):
        msg = (
            f"y0 must have {len(model.state_names)} values, one per state, got shape {start.shape}"
        )
        raise ValueError(msg)

    def rhs(y: FloatArray, ti: float) -> Sequence[Any]:
        return model.derivatives(ti, y, params, inputs)

    # odeint reports failure as a warning and returns whatever it reached; the
    # message is raised as an IntegrationError instead.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ODEintWarning)
        states, info = odeint(rhs, start, times, full_output=True)
    if info["message"] != "Integration successful.":
        msg = f"{model.name}: {info['message']}"
        raise IntegrationError(msg)

    return Trajectory(
        model=model,
        params=params,
        inputs=inputs,
        t=times,
        states=states,
        outputs=dict(model.outputs(times, states, params, inputs)),
    )
