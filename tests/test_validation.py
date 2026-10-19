"""The model against `data/output.csv`, reference KSS from the original authors' model.

The file holds 48 hours of light (melanopic EDI, lux) and KSS at 10-second
steps for a wake 08:00 to 20:00 protocol. Its provenance is recorded in
`docs/research/2026-10-19-circadian-model-landscape.md`. `data/` is not
committed, so this test runs only where the file is present.

With the model's own initial state the KSS error while awake was 0.285 when
this test was written (0.146 after 14 days of entrainment, which suggests the
reference started from an entrained state). The threshold guards against a
change making the agreement worse; a science fix that improves it should
tighten the threshold in the same pull request.
"""

from pathlib import Path

import numpy as np
import pytest

from arousal_dynamics import ArousalModel, Inputs, LightInput, LightUnit, WakeSchedule, simulate

REFERENCE = Path(__file__).parents[1] / "data" / "output.csv"
AWAKE_RMSE_LIMIT = 0.30

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


def test_kss_tracks_the_reference_while_awake() -> None:
    t, lux, reference = _read_reference()
    light = LightInput.from_samples(t, lux, unit=LightUnit.MELANOPIC_EDI, interpolation="previous")
    run = simulate(ArousalModel(), Inputs(light, WakeSchedule(waketime=8, bedtime=20)), t)
    clock = t / 3600 % 24
    awake = (clock >= 8) & (clock <= 20)
    rmse = float(np.sqrt(np.mean((run["KSS"][awake] - reference[awake]) ** 2)))
    assert rmse <= AWAKE_RMSE_LIMIT
