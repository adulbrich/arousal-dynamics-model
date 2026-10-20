# 0002. Keep the arousal model to the light it was fitted on

The arousal model's light responses were fitted to laboratory exposures against a dim
background: melatonin suppression to evening and night-time exposures only (T20 Table
2), the alerting effect to five evening and night-time studies and two daytime studies
that compare bright light with dim light (T20 Table 3), and phase resetting to phase
response curves (T20 Table 1). No fit compares daylight with office light. As
implemented, the alerting term (T20 Eq 14 with S_b = 0.05 W/m² and S_c = 1/223.5 W/m²,
audit D3) is saturated above about 60 lux melanopic EDI, and its effect on KSS is capped
at |ν_la|·H, about 1.4 points, so an outdoor day and an office day give nearly the same KSS: under 0.1 point apart on
the office schedule of `tests/conftest.py` run for 28 days with 250 or 10,000 lux from
08:00 to 18:00. We keep the model as it is, equations and constants, and leave daytime
light to a separate model behind the `Model` interface (ADR 0001), because no daytime
dataset exists to fit against and because constants alone cannot lift the cap.

## Considered options

- **Refit S_b and S_c for daytime.** It moves where the alerting effect saturates but
  not how large it can get, and there is nothing to fit against.
- **Change the alerting equation** (a sigmoid on log irradiance, a larger ν_la). That is
  a new model under the old name, and its results would no longer be comparable with the
  papers'.

## Consequences

- Runs with daytime light above about 60 lux melanopic EDI are outside the model's
  evidence. Differences between bright daytime conditions are not predictions.
- Circadian phase still responds to daytime light, because the fitted phase response
  curves include bright pulses at every circadian phase, but only against a dim
  background.
- Audit E3, whether the code reproduces T20's own alerting dose-response (half the
  maximum effect at 0.05 W/m², saturation at 0.86 W/m², T20 Section 3.3.4), stays open.
  It is a reproduction question inside this model's scope and waits with D1 and D2.
- A daytime model should not inherit the photic drive as printed. P18 Table 1 Eq 13 and
  T20 Eq 6 and 7, which the code follows, place I0 = 100 lux and I1 = 9,500 lux the
  other way round from St Hilaire et al. 2007 Eq 9, the Kronauer-family form they build
  on: α0·(I/9500)^0.5·I/(I + 100). With the same α0 the printed form is about 0.2 times
  St Hilaire's at 100 lux, equal near 1,000 lux and about 5 times at 10,000 lux. Neither
  paper says whether the swap is deliberate.

## Studies a daytime model starts from

- **Alerting has no plateau in the daytime data.** KSS falls 0.63 points per decade of
  illuminance at the eye from 20 to 2,000 lux, a weak fit (R² 0.04) with no plateau
  found (Smolders et al. 2018, doi:10.1177/0748730418796438). In everyday life, KSS falls about 0.5 points per decade
  of melanopic EDI at the wrist, over a range that includes outdoor light, with
  individual slopes from -1.6 to 0.8 (Didikoglu et al. 2023, doi:10.1073/pnas.2301608120;
  open data, doi:10.48420/23786238).
- **The effect is gated.** Well-rested, dim-adapted people showed no daytime alerting
  effect on KSS from 24 to 2,000 lux (Lok et al. 2018, doi:10.1177/0748730418796036),
  while studies after short sleep or a dim morning found one (Phipps-Nelson et al. 2003,
  doi:10.1093/sleep/26.6.695; Rüger et al. 2006, doi:10.1152/ajpregu.00121.2005).
- **Daylight advances the clock.** A week of camping under natural light only moved
  melatonin onset about 2 h earlier in summer (Wright et al. 2013,
  doi:10.1016/j.cub.2013.06.039) and 2.6 h earlier in winter (Stothard et al. 2017,
  doi:10.1016/j.cub.2016.12.041).
- **Earlier light changes the response to later light.** After three days at 1 lux
  rather than 90 lux, evening light suppressed melatonin by 86.1 % rather than 51.3 % and
  delayed the clock 1.67 h rather than 1.03 h (Chang et al. 2011,
  doi:10.1113/jphysiol.2010.201194). This model has no light-history term.
- **Bright days raise nocturnal melatonin.** A week of days at about 5,000 lux, after a
  week of dim days, raised the peak of the nocturnal melatonin rise without moving its
  onset further (Takasu et al. 2006, doi:10.1152/ajpregu.00211.2006).
