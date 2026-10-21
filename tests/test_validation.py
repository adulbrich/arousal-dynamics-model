"""The model against `data/output.csv`, KSS from the original authors' model.

The file holds 48 hours of light (melanopic EDI, lux) and KSS at 10-second
steps. It is a spreadsheet export with no history in git; `data/` is not
committed, so these tests run only where the file is present.

The light is 168.6 lux from 08:00 to 20:00 and 27.0 lux otherwise, each switch
ramped over about six minutes; on day 1 from 09:00 to 16:00 it is instead the
profile in `data/input_irradiance_mel.csv`, at about 1.1 times its
`illuminance_mel`. The two levels are the 250 and 40 lux of the entrainment
protocol of Postnova et al. 2018 (P18, p. 204) times 0.6745812. The
collaborator who sent them in 2021 gave them as melanopic irradiance at the
CIE D65 efficacy (0.036 and 0.22 W/m^2, commit b048547), which is how this
test converts them.

The KSS fixes the rest of the protocol:

- No forced wake. The alerting response starts at a spontaneous wake at 08:13
  (08:14 on day 2), not at the 08:00 lights-on, and sleep begins at 23:42 both
  nights.
- An entrained start with Theta_L = 0. Day 1 tracks day 2 except for a 0.17
  point excess at 00:00 that decays with Theta_L's 24-minute time constant.
  Tekieh et al. 2020 (p. 7) start every simulation entrained, with sleep from
  23:30 to 08:00; the run here starts from 60 days on the P18 protocol.

The two parameter sets score, as KSS error while awake (08:00 to 20:00) and
over the 48 hours:

- The defaults, with S_c = 0.0306 W/m^2 and theta_H = 2.284 fitted to this
  file: 0.015 and 0.019.
- `ArousalParameters.papers()`, with the printed S_c = 1/223.5 W/m^2 and
  theta_H = 2.28: 0.112 and 0.120. Its alerting term is a switch between
  about 25 and 55 lux, so KSS reads up to 0.48 too low in the 42 to 72 lux
  of the day 1 morning and 0.25 too high in the 27 lux evenings.

On the same protocol, the code before the fixes for issues #6, #11 and #14
scored 0.177 and 0.343. With its own initial state and forced wake from 08:00
to 20:00, the test's protocol until then, it scored 0.285 while awake.

What the fit leaves is the first wake. From the P18 start the model wakes at
08:10, 2.8 minutes before the file, which makes the largest single error: 0.15
point for those minutes. Its sleep onsets and second wake fall within 0.7
minutes of the file's.

The limits guard against a change making the agreement worse; a science fix
that improves it should tighten them in the same pull request.
"""

from pathlib import Path

import numpy as np
import pytest

from arousal_dynamics import (
    ArousalModel,
    ArousalParameters,
    Inputs,
    LightInput,
    LightUnit,
    Trajectory,
    simulate,
)

REFERENCE = Path(__file__).parents[1] / "data" / "output.csv"
# (RMSE limit while awake, RMSE limit over 48 hours) for each parameter set
LIMITS = {"defaults": (0.02, 0.025), "papers": (0.13, 0.14)}
PARAMETER_SETS = {"defaults": ArousalParameters(), "papers": ArousalParameters.papers()}

pytestmark = [
    pytest.mark.validation,
    pytest.mark.slow,
    pytest.mark.skipif(not REFERENCE.exists(), reason=f"{REFERENCE} is not present"),
]


def _read_reference() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time in seconds, light in lux, and KSS, from the semicolon-separated, comma-decimal file."""
    rows = REFERENCE.read_text(encoding="utf-8").replace(",", ".").splitlines()[1:]
    data = np.array([[float(v.strip('"')) for v in row.split(";")] for row in rows])
    lux, kss, hours = data.T
    return hours * 3600.0, lux, kss


@pytest.fixture(scope="module")
def reference() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _read_reference()


@pytest.fixture(scope="module", params=sorted(PARAMETER_SETS))
def run(
    request: pytest.FixtureRequest,
    reference: tuple[np.ndarray, np.ndarray, np.ndarray],
    p18_entrainment: Trajectory,
) -> tuple[str, Trajectory]:
    t, lux, _ = reference
    start = p18_entrainment.states[-1].copy()
    start[ArousalModel.state_names.index("Theta_L")] = 0.0
    light = LightInput.from_samples(t, lux, unit=LightUnit.MELANOPIC_EDI, interpolation="previous")
    params = PARAMETER_SETS[request.param]
    return request.param, simulate(
        ArousalModel(), Inputs(light, schedule=None), t, params=params, y0=start
    )


def _rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(error**2)))


def test_kss_tracks_the_reference_while_awake(
    reference: tuple[np.ndarray, np.ndarray, np.ndarray], run: tuple[str, Trajectory]
) -> None:
    t, _, kss = reference
    name, trajectory = run
    clock = t / 3600 % 24
    awake = (clock >= 8) & (clock <= 20)
    assert _rmse(trajectory["KSS"][awake] - kss[awake]) <= LIMITS[name][0]


def test_kss_tracks_the_reference_over_48_hours(
    reference: tuple[np.ndarray, np.ndarray, np.ndarray], run: tuple[str, Trajectory]
) -> None:
    _, _, kss = reference
    name, trajectory = run
    assert _rmse(trajectory["KSS"] - kss) <= LIMITS[name][1]
