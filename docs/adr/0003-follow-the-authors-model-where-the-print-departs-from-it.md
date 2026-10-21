# 0003. Follow the authors' model where the print departs from it

Where the papers' own account of the model's behaviour, or the KSS the original authors
shared (`data/output.csv`), shows that their model does something other than what the
papers print, the code follows the model. Three such fixes have landed, issue by issue: the
photic drive takes St Hilaire et al. 2007's form (issue #6), the alerting term sees light
only while awake (issue #14), and its sigmoid is normalized to [0, 1] (Tekieh et al. 2020,
T20, p. 7) and refit to a width S_c of 0.0306 W/m² (issue #11). This amends ADR 0002,
which kept the equations and constants as they were and counted gating the alerting term
by sleep among the changes that make a different model. These changes are confined to
reproducing the authors' night-time model, which ADR 0002 keeps in scope. Its decision
on daytime light stands.

## Considered options

- **Keep the printed equations and offer the authors' behaviour as overrides.** The
  printed photic form entrains the model to sleep from 22:30 to 07:00 on the 250/40/0 lux
  protocol, where Postnova et al. 2016 (p. 500) and 2018 (p. 204) and T20 (p. 7) report
  sleep from 23:30 to 08:00. A default that follows the print would match neither the
  papers' account nor the authors' output.
- **Also refit theta_H.** theta_H = 2.284 removes the last constant 0.05 KSS point
  against the shared output and rounds to the printed 2.28. But rounding can absorb any
  residual under 0.06, so the evidence is weak. It stays as printed, and
  `tests/test_validation.py` reports the offset.

## Consequences

- S_c is the one constant without a paper behind it; its field says so and names the
  fit. The sigmoid now reaches 0.99 at 0.20 W/m², about 150 lux melanopic EDI, instead of
  about 55 lux. The alerting term still saturates at office light, and its effect is
  still capped at |nu_la| H.
- On its protocol the model reproduces the shared KSS to 0.049 point while awake and
  0.056 over 48 hours, against 0.177 and 0.343 before (`tests/test_validation.py`).
- ADR 0002's measurement under the fixed model: on the office schedule of
  `tests/conftest.py` run for 28 days with 250 or 10,000 lux melanopic EDI from 08:00 to
  18:00, KSS between 08:00 and 18:00 over the last three days differs by 0.07 point on
  average and 0.11 at most (ADR 0002 measured 0.08 and 0.16). Circadian phase responds
  less to daylight than before: St Hilaire's form gives 10,000 lux about 8 times the
  photoreceptor drive of 250 lux, where the printed form gave about 100 times. The daily
  minimum of X comes 0.8 hours earlier under 10,000 lux than under 250 lux, against 1.3
  hours before.
- The comparison rests on `data/output.csv` being the authors' model output, which is
  how the owner received it. If that is wrong, the S_c refit and the evidence for gating
  go with it. The fix to the photic form stands on the papers alone.
- Untouched: melatonin suppression still sees ungated light (the rest of issue #14), and
  issues #8, #9, #12 and #13 stay open. The shared output cannot test them: melatonin
  does not reach KSS, and its protocol forces no wake.
