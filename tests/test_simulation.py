"""`simulate`, `Trajectory` and the `Model` interface."""

import sys
from collections.abc import Mapping
from dataclasses import fields, replace

import numpy as np
import pytest

from arousal_dynamics import (
    ArousalModel,
    ArousalParameters,
    Inputs,
    IntegrationError,
    LightInput,
    LightUnit,
    Model,
    Trajectory,
    WakeSchedule,
    simulate,
    time_grid,
)
from arousal_dynamics.units import FloatArray

ONE_DAY = time_grid(days=1, steps=145)


@pytest.fixture
def inputs() -> Inputs:
    return Inputs(LightInput.constant(100, unit=LightUnit.MELANOPIC_EDI), WakeSchedule(7, 23))


class Decay:
    """dy/dt = -k y + light: a model with a closed-form answer, to test the seam."""

    name = "decay"
    state_names = ("y",)
    time_unit = "s"

    def default_parameters(self) -> float:
        return 0.5

    def initial_state(self, params: float) -> FloatArray:
        return np.array([1.0])

    def derivatives(self, t: float, y: FloatArray, params: float, inputs: Inputs) -> list[float]:
        return [-params * y[0] + float(inputs.light(t))]

    def outputs(
        self, t: FloatArray, states: FloatArray, params: float, inputs: Inputs
    ) -> Mapping[str, FloatArray]:
        return {"twice": 2 * states[:, 0]}


class Blowup(Decay):
    """dy/dt = y^2 from y = 1 diverges at t = 1."""

    name = "blowup"

    def derivatives(self, t: float, y: FloatArray, params: float, inputs: Inputs) -> list[float]:
        return [y[0] ** 2]


class TestModelInterface:
    def test_arousal_model_satisfies_it(self) -> None:
        assert isinstance(ArousalModel(), Model)

    def test_a_second_model_runs_through_simulate(self) -> None:
        t = np.linspace(0.0, 10.0, 21)
        run = simulate(Decay(), Inputs(LightInput.constant(0.0)), t)
        np.testing.assert_allclose(run["y"], np.exp(-0.5 * t), rtol=1e-6)
        np.testing.assert_allclose(run["twice"], 2 * run["y"])

    def test_parameters_reach_the_model(self) -> None:
        t = np.linspace(0.0, 4.0, 9)
        run = simulate(Decay(), Inputs(LightInput.constant(0.0)), t, params=2.0)
        # odeint's default absolute tolerance is 1.5e-8.
        np.testing.assert_allclose(run["y"], np.exp(-2.0 * t), rtol=1e-6, atol=1e-8)

    def test_inputs_reach_the_model(self) -> None:
        # Steady state of dy/dt = -k y + E is E / k.
        t = np.linspace(0.0, 60.0, 7)
        run = simulate(Decay(), Inputs(LightInput.constant(3.0)), t)
        assert run["y"][-1] == pytest.approx(6.0, rel=1e-6)

    def test_solver_failure_raises(self) -> None:
        with pytest.raises(IntegrationError, match="blowup"):
            simulate(Blowup(), Inputs(LightInput.constant(0.0)), np.linspace(0.0, 2.0, 5))


class TestSimulate:
    def test_reports_at_the_requested_times(self, inputs: Inputs) -> None:
        run = simulate(ArousalModel(), inputs, ONE_DAY)
        np.testing.assert_array_equal(run.t, ONE_DAY)
        assert run.states.shape == (ONE_DAY.size, 9)

    def test_starts_from_the_given_state(self, inputs: Inputs) -> None:
        y0 = np.array([-4.0, -0.1, 13.0, -0.1, -1.0, 0.1, 0.0, 0.0, 20.0])
        run = simulate(ArousalModel(), inputs, ONE_DAY, y0=y0)
        np.testing.assert_array_equal(run.states[0], y0)

    def test_default_parameters_are_the_default(self, inputs: Inputs) -> None:
        a = simulate(ArousalModel(), inputs, ONE_DAY)
        b = simulate(ArousalModel(), inputs, ONE_DAY, params=ArousalParameters())
        np.testing.assert_array_equal(a.states, b.states)

    def test_a_parameter_override_changes_the_run(self, inputs: Inputs) -> None:
        base = simulate(ArousalModel(), inputs, ONE_DAY)
        refit = simulate(
            ArousalModel(), inputs, ONE_DAY, params=replace(ArousalParameters(), s_b=0.3)
        )
        assert not np.allclose(base["Theta_L"], refit["Theta_L"])
        # The light term feeds KSS but not the sleep/wake switch.
        np.testing.assert_array_equal(base["S"], refit["S"])

    @pytest.mark.parametrize(
        ("t", "message"),
        [
            ([0.0], "at least two"),
            ([[0.0, 1.0]], "1-D"),
            ([0.0, 10.0, 10.0], "strictly increasing"),
            ([10.0, 0.0], "strictly increasing"),
        ],
    )
    def test_rejects_bad_time_grids(self, inputs: Inputs, t: list, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            simulate(ArousalModel(), inputs, t)

    def test_rejects_a_wrong_initial_state(self, inputs: Inputs) -> None:
        with pytest.raises(ValueError, match="9 values"):
            simulate(ArousalModel(), inputs, ONE_DAY, y0=[0.0, 1.0])


class TestTrajectory:
    @pytest.fixture
    def run(self, inputs: Inputs) -> Trajectory:
        return simulate(ArousalModel(), inputs, ONE_DAY)

    def test_names_cover_states_then_outputs(self, run: Trajectory) -> None:
        assert run.names[:9] == ArousalModel.state_names
        assert set(run.names[9:]) == {"light", "S", "C", "KSS", "phi", "aMT6s"}

    def test_states_index_their_column(self, run: Trajectory) -> None:
        np.testing.assert_array_equal(run["H"], run.states[:, 2])

    def test_unknown_name(self, run: Trajectory) -> None:
        with pytest.raises(KeyError, match="kss"):
            run["kss"]

    def test_hours(self, run: Trajectory) -> None:
        assert run.t_hours[-1] == pytest.approx(24.0)

    def test_dataframe(self, run: Trajectory) -> None:
        frame = run.to_dataframe()
        assert list(frame.columns) == list(run.names)
        assert frame.index.name == "t"
        np.testing.assert_array_equal(frame["KSS"].to_numpy(), run["KSS"])

    def test_dataframe_without_pandas(
        self, run: Trajectory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "pandas", None)
        with pytest.raises(ImportError, match=r"arousal-dynamics\[pandas\]"):
            run.to_dataframe()


def test_every_parameter_names_its_unit_and_source() -> None:
    for f in fields(ArousalParameters):
        assert "unit" in f.metadata, f.name
        assert f.metadata["source"], f.name
