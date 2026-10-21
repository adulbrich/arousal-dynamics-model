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

With the model's defaults the KSS error is 0.049 while awake (08:00 to 20:00)
and 0.056 over the 48 hours. On the same protocol the code before the fixes for
issues #6, #11 and #14 scored 0.177 and 0.343 (0.285 while awake from its own
initial state with forced wake from 08:00 to 20:00, this test's protocol until
then). Two residuals remain:

- An offset: the model reads 0.05 point lower throughout, awake and asleep.
  theta_H = 2.284 instead of the 2.28 of P18 Table 3, a value that rounds to
  the printed one, removes it (0.015 while awake, 0.019 over 48 hours). The
  default stays as printed.
- The first wake. From the P18 start the model wakes at 08:10, 2.8 minutes
  before the file, which makes the largest single error, 0.20 point for those
  minutes. Its sleep onsets and second wake fall within 0.7 minutes of the
  file's.

The limits guard against a change making the agreement worse; a science fix
that improves it should tighten them in the same pull request.
"""

from pathlib import Path

import numpy as np
import pytest

from arousal_dynamics import (
    ArousalModel,
    Inputs,
    LightInput,
    LightUnit,
    Trajectory,
    simulate,
)

REFERENCE = Path(__file__).parents[1] / "data" / "output.csv"
AWAKE_RMSE_LIMIT = 0.06
RMSE_LIMIT = 0.07

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


@pytest.fixture(scope="module")
def run(
    reference: tuple[np.ndarray, np.ndarray, np.ndarray], p18_entrainment: Trajectory
) -> Trajectory:
    t, lux, _ = reference
    start = p18_entrainment.states[-1].copy()
    start[ArousalModel.state_names.index("Theta_L")] = 0.0
    light = LightInput.from_samples(t, lux, unit=LightUnit.MELANOPIC_EDI, interpolation="previous")
    return simulate(ArousalModel(), Inputs(light, schedule=None), t, y0=start)


def _rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(error**2)))


def test_kss_tracks_the_reference_while_awake(
    reference: tuple[np.ndarray, np.ndarray, np.ndarray], run: Trajectory
) -> None:
    t, _, kss = reference
    clock = t / 3600 % 24
    awake = (clock >= 8) & (clock <= 20)
    assert _rmse(run["KSS"][awake] - kss[awake]) <= AWAKE_RMSE_LIMIT


def test_kss_tracks_the_reference_over_48_hours(
    reference: tuple[np.ndarray, np.ndarray, np.ndarray], run: Trajectory
) -> None:
    _, _, kss = reference
    assert _rmse(run["KSS"] - kss) <= RMSE_LIMIT
