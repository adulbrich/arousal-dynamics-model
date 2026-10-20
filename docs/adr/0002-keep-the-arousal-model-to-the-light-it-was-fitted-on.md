# 0002. Keep the arousal model to the light it was fitted on

The arousal model is poor at high light levels: its light responses were calibrated
mostly on evening and night-time studies, with few daytime data, and its alerting term
saturates at office light, so an outdoor day and an office day give nearly the same KSS.
We keep its equations and constants as they are for now, because we have no daytime
dataset of our own to fit against and the daytime evidence calls for structure this
model lacks; a daytime-capable model, if one is built, will be a separate model behind
the `Model` interface (ADR 0001).

## Considered options

- **Refit S_b and S_c for daytime.** The alerting sigmoid (Tekieh et al. 2020, T20, Eq
  14) was taken directly from a dose-response curve pooled from eight laboratory and
  field studies, four of them in the daytime and only two of those measured against dim
  light (Hommes and Giménez 2015, Table 1 and Discussion, via T20 p. 7). As coded (S_c =
  1/223.5 W/m², audit D3) it saturates above about 60 lux melanopic EDI; T20 reports its
  own default saturating at 0.86 W/m², about 650 lux melanopic EDI (T20 Section 3.3.4,
  audit E3). A refit moves where the effect saturates, not how large it can get, and a
  logistic in linear irradiance cannot be graded across the decades between office light
  and daylight.
- **Raise ν_la.** The light term's effect on KSS is capped at |ν_la|·H, about 1.4
  points. ν_la was fitted to seven data sets, five in the evening or at night and two in
  the daytime comparing bright light with dim light (T20 Table 3); raising it for daytime
  changes the predictions at night too.
- **Change the alerting equation** (a sigmoid on log irradiance, light history over
  days, gating by sleep state). That is a different model under the same name: the
  separate model this record defers.

## Consequences

- Runs with daytime light above office levels are outside the model's evidence. On the
  office schedule of `tests/conftest.py` run for 28 days with 250 or 10,000 lux
  melanopic EDI from 08:00 to 18:00, KSS between 08:00 and 18:00 over the last three days
  differs by 0.08 point on average and 0.16 at most (5-minute steps), none of it through
  the light term, which is saturated in both.
- Circadian phase still responds to daytime light. The oscillator's light response was
  calibrated before T20 and checked against phase response curves that include bright
  pulses at every circadian phase (T20 Section 2.3.1 and Table 1), all against a dim
  background.
- Audit E3, whether the code reproduces T20's own alerting dose-response, stays open as
  a reproduction question inside this model's scope, and waits with D1 and D2.
- The daytime evidence points to structure this model lacks: an alerting response that
  may keep growing across decades of light (about 0.5 KSS point per decade of melanopic
  EDI in everyday life, Didikoglu et al. 2023, doi:10.1073/pnas.2301608120, open data
  doi:10.48420/23786238; weak in the laboratory, Smolders et al. 2018,
  doi:10.1177/0748730418796438), light history over days rather than the hours of the
  photoreceptor state P (Postnova et al. 2018, Table 1 Eq 6; Chang et al. 2011,
  doi:10.1113/jphysiol.2010.201194), an alerting effect that well-rested people may not
  show (Lok et al. 2018, doi:10.1177/0748730418796036), and a melatonin response whose
  direction depends on when the light comes (Takasu et al. 2006,
  doi:10.1152/ajpregu.00211.2006; Lazar et al. 2025, doi:10.1038/s44323-025-00040-6).
