# Equation audit: models.py and constants.py against the source papers

Date: 2026-10-19. Code audited: commit `55eeb3e` ("added additional plots"), read with `git show`, not from the working tree. No code was changed.

Papers (local PDFs):

- **P16**: Postnova et al. 2016, J Biol Rhythms 31(5). Journal page numbers (Eqs 1-3 on p. 500 = PDF p. 3).
- **P18**: Postnova et al. 2018, J Biol Rhythms. Journal page numbers (Table 1 on p. 206 = PDF p. 4; Table 3 and Eq 23 on p. 209 = PDF p. 7).
- **A18**: Abeysuriya et al. 2018, J Pineal Res, accepted-manuscript PDF. PDF page numbers (Eq 1 p. 4; Eqs 2-4 p. 5; steady-state text p. 8; Eqs 14-15 p. 9; Table 1 p. 25; Fig 2 p. 28).
- **T20**: Tekieh et al. 2020, J Pineal Res. Page numbers are the printed "x of 16", equal to PDF pages.
- Phillips 2019: no constant or docstring points to it; grepped only, not read.

Every equation and value below was transcribed from a rendered PDF page image, except where marked "not read". Numerical checks were run on scratch copies of `models.py` and `constants.py` from `55eeb3e` with the project `.venv` (numpy 2.5.4, scipy 1.18.1); nothing in the repo was modified by them.

---

## 1. Summary

- **Constants**: 51 in `constants.py`. 44 match the paper with no caveat. 7 are flagged: `S_C`, `TAU_A`, `TAU_ALPHA`, `V_WE`, and `R_A`, `R_B`, `R_C` (values correct, used in the wrong functional form).
- **Functions and equations**: 27 rows. 21 match. 2 are code bugs. 2 match in form but carry a flagged parameter (`sigmoid` via `S_C`, `dA/dt` via `TAU_ALPHA`). 2 have no paper source (`irradiance`, `y0`).

### Code bug

- **D1** `circadian_phase` uses two-quadrant `atan(Y / X)`; the papers' phase spans -pi..pi and needs four-quadrant `arctan2(Y, X)`. Melatonin synthesis runs about 1 h/day instead of about 8 h/day.
- **D2** `melatonin_suppression` combines A18 Eq 4's logistic form with T20 Eq 9's Hill-form parameters. Gives r = 0.509 in darkness; both papers require r = 1.

### Documented deviation

- **D3** `S_C = 1.0 / 223.5` W/m^2 instead of the printed 223.5 m^2/W. Documented in the constant comment and in commit 652588a. The correction is dimensionally forced (see E2), but see E3 for the value.

### Paper error

- **E1** P18 Table 1 Eq 17 and A18 Eq 3 print `atan(Y/X)` for what must be a four-quadrant arctangent (P18 gives phi_crit = -2.98, outside atan's range; A18 says the phase wraps from -pi to +pi).
- **E2** T20 p. 7 prints S_c = 223.5 m^2/W, which makes the exponent of Eq 14 dimensional. The same line also prints S_b's unit twice.
- **E3** T20 Fig 6 (p. 12) shows a default-model dose-response far shallower than Eq 14 with S_c = 1/223.5 W/m^2 allows. Neither reading of the printed S_c reproduces the figure; the figure implies S_c on the order of 0.02 W/m^2. Which value the authors ran is unclear.
- **E4** Minor typos that do not reach the code: P18 Table 3 KSS Theta_0 alternative value printed as (22.43), positive; T20 p. 8 prints the tau_L range as (0.24) minutes.

### Unclear

- **U1** `TAU_ALPHA` = 3.11 h (A18 Table 1) drives dA/dt, while T20 p. 7 resets tau_A to 1.5 h together with the Hill-form suppression whose parameters the code uses. `TAU_A` = 1.5 h is defined but never used.
- **U2** `V_WE` = -0.07 mV matches P18 Table 1, but P16 p. 507 states the new default is V_WE = 0.07 mV.
- **U3** Light is gated by sleep state (`IE * S`) only inside `photoreceptor_conversion_rate`; `melatonin_suppression(IE)` and `sigmoid(IE)` see ungated light. The papers do not say either way for r and S(E).

---

## 2. Discrepancies in detail

### D1. `circadian_phase`: two-quadrant arctangent (Code bug)

**Code** (`circadian_phase`, models.py):

```python
phi = atan(Y / X)
```

Docstring: "Table 1 Equation 17 in [Postnova et al. 2018]". Consumer (`melatonin_synthesis_regulation`, unchanged and correct):

```python
if constants.PHI_ON <= phi <= constants.PHI_OFF:
    m_phi = 0
else:
    m_phi = 1
```

**Papers**

- A18 Eq 3, PDF p. 5: `phi = atan(Y / X)`. The text that follows states that this phase falls over time and normally jumps from -pi to +pi during sleep.
- A18 Eq 2, PDF p. 5: `m(phi) = 1 if phi < phi_on or phi > phi_off; 0 if phi_on <= phi <= phi_off`. The text says phi_on and phi_off bound the phases at which synthesis is active, and with a decreasing phase Eq 2 puts synthesis ON outside [SynOn, SynOff] and OFF inside it.
- A18 Fig 2b, PDF p. 28: phase axis labelled -pi, -pi/2, 0, pi/2, pi. Phase falls linearly through the day, crosses phi_on about 20:20 (SynOn), wraps from -pi to +pi near 02:00, and crosses phi_off about 03:45 (SynOff). The shaded synthesis band is phi < phi_on and phi > phi_off.
- A18 Table 1, PDF p. 25: phi_on = -1.44 rad, phi_off = 2.78 rad.
- P18 Table 1 Eq 17, p. 206: `phi_crit = -2.98 = atan(Y / X)`.

**Commit**: `atan(Y / X)` was introduced in 76de415 (2025-05-02) "added u(t) and new notebook to run". The Postnova Eq 17 docstring citation was added in acd6931 (2025-05-08) "added equations and finding scenarios". No comment or message discusses the arctangent range.

**Evidence**

- `numpy.atan` returns values in (-pi/2, pi/2). phi_off = 2.78 and P18's phi_crit = -2.98 both lie outside that range, so neither paper can mean the one-argument function.
- The ODEs give X' proportional to Y and Y' proportional to -X, so (X, Y) rotates clockwise and `arctan2(Y, X)` decreases monotonically, matching A18's description. In the scratch run (below), the `arctan2` phase decreased on every step.
- With `atan`, `phi < -1.44` holds only for phi in (-pi/2, -1.44), and the 2-quadrant fold hits that window twice per cycle.
- Scratch run, using code from 55eeb3e: 14 days; light at 250 lx FL4100K-equivalent from 08:00 to 20:00, 40 lx from 20:00 to 23:00, dark otherwise, converted with F_4100K; statistics over the last 2 days. Sleep ran 22:25 to 06:05.
  - As committed (`atan`): m = 1 for 0.96 h/day, split across two windows, about 08:45 to 09:15 and 20:45 to 21:15. Peak A = 0.075, peak rho_b = 22.3 pmol/L.
  - With `arctan2(Y, X)` swapped in: m = 1 from 20:45 to 04:40 (7.96 h/day), close to A18 Fig 2. Peak rho_b = 151.5 pmol/L, still low because of D2.
- The initial state shows the same problem: y0 has X = -0.14, Y = -1.07, giving `arctan2` = -1.70 rad (synthesis on, as expected at midnight) but `atan` = +1.44 rad (synthesis off).
- Also, at X = 0 `Y / X` divides by zero; `arctan2` does not.

**Recommendation**: fix. Use `numpy.arctan2(Y, X)` (argument order Y, X). Keep `melatonin_synthesis_regulation` as is; its inside/outside logic already matches A18 Eq 2. Re-point the docstring to A18 Eq 3, noting the four-quadrant reading (see E1).

---

### D2. `melatonin_suppression`: logistic form with Hill-form parameters (Code bug)

**Code** (`melatonin_suppression`, models.py):

```python
r = 1 - constants.R_A / (1 + exp( - (E_emel - constants.R_B) / constants.R_C))
```

with `R_A = 1`, `R_B = 0.031  # W/m^2`, `R_C = 0.82`. Docstring: "Equation 4 in [Abeysuriya et al. 2018]".

**Papers**

- T20 Eq 9, p. 7:

  ```
  r(E_e,mel) = 1 - r_a / (1 + (E_e,mel / r_b)^(-r_c))
  ```

  with r_a = 1, r_b = 0.031 W/m^2 and r_c = 0.82, fit to Prayag et al. data. This is a Hill function in E; r_c is a dimensionless exponent.
- A18 Eq 4, PDF p. 5:

  ```
  r(I) = 1 - r_a * 1 / (1 + exp(-(I - r_b) / r_c))
  ```

  where I is photopic illuminance in lux. A18 Table 1, PDF p. 25: r_a = 0.94, r_b = 110 lx, r_c = 26 lx.
- A18 PDF p. 8: the steady state is derived with m = 1 and, for darkness, r = 1.

**Commits**

- 8fa458a (2021-02-20) "corrections and clean up": introduced the T20 Hill form, `r = 1 - (r_a/(1+power(E_emel/r_b,-r_c))) # Tekieh et al. 2020 - Equation 9`.
- acd6931 (2025-05-08): added the docstring "Equation 4 in [Abeysuriya et al. 2018]" while the code was still the T20 Hill form, so the citation was wrong first.
- 4ca72b2 (2025-06-25) "fixed problem in melatonin suppression, sigmoid, and made model more flexible": replaced the Hill form with the A18 logistic, keeping the T20 constants. The message does not say what the problem was.
  - The same commit added `return max(interpolator(t),0)` to `irradiance()`. Before it, the cubic interpolator's output was unclamped, and the Hill form returns NaN for negative input (checked: `np.power(-1e-6/0.031, -0.82)` is `nan`). That is a plausible cause of the "problem". This is an inference, not documented.
  - With the clamp in place, the Hill form only meets E = 0, where `np.power(0.0, -0.82)` is `inf` with a RuntimeWarning and r evaluates to exactly 1.0.

**Evidence**

| E (W/m^2) | code (logistic, T20 params) | T20 Eq 9 (Hill) | A18 Eq 4 with A18 params, E converted to FL4100K lux |
|---|---|---|---|
| 0 | 0.5095 | 1.0000 | 0.9865 |
| 0.031 | 0.5000 | 0.5000 | 0.9448 |
| 0.1 | 0.4790 | 0.2768 | 0.4226 |
| 1.0 | 0.2347 | 0.0548 | 0.0600 |

The code matches neither paper. In darkness it caps synthesis at 51% of peak, contradicting both papers (r = 1 with no light). It also treats `R_C` = 0.82 as a width in W/m^2, while T20 defines it as a dimensionless exponent.

Scratch run (same scenario as D1, both with `arctan2`): peak rho_b was 151.5 pmol/L with the committed logistic and 283.3 pmol/L with the T20 Hill form. The paper's plasma peak rho_b* is 325 pmol/L.

**Recommendation**: fix. Restore T20 Eq 9 with an explicit guard (`E <= 0` returns 1.0) and keep the clamp in `irradiance()`. Re-point the docstring to T20 Eq 9. Using A18's logistic instead would require A18's photopic-lux parameters and a lux input, which conflicts with the melanopic (2020) model the code runs.

---

### D3. `S_C = 1.0 / 223.5` (Documented deviation; value questioned by E3)

**Code** (`constants.py`):

```python
S_B = 0.05  # W/m^2
S_C = 1.0 / 223.5  # W/m^2 (typo corrected from paper)
```

used in `sigmoid`:

```python
S = 1 / (1 + exp((S_B - E_emel) / S_C))
```

**Paper**

- T20 Eq 14, p. 7: `S(E_e,mel) = 1 / (1 + exp((S_b - E_e,mel) / S_c))`.
- T20 p. 7, values as printed: S_b = 0.05 W/m2 W/m^2 (unit printed twice) and S_c = 223.5 m^2/W.

**Commits**

- 8fa458a (2021-02-20): `S_c = 223.5 # m^2/W`, as printed.
- 652588a (2021-05-17) "changed to correct parameters": `S_c = 1/223.5 # m^2/W, there is a typo in the paper, it should be S_c^{-1} = 223.5 m^2/W`.
- b3c14a6 (2025-05-01) moved it to `constants.py`. 4ca72b2 (2025-06-25) changed the comment to "W/m^2 (typo corrected from paper)".
- ed3e0e1 (2025-07-17) "more plots" temporarily set `S_B = 0.3`, `S_C = 0.01604672` (and a commented `NU_LA = -1.619739`). d10f2f2 (2025-07-17) restored the paper-derived values.
- Earlier, 8395c8d (2021-04-14) divided E by 0.0013262 to convert to illuminance; 33089b3 (2021-05-13) removed that conversion with the comment "the parameters were then computed for irradiance so we can use irradiance directly".

**Evidence**

- In Eq 14, (S_b - E) has units W/m^2. Dividing by S_c in m^2/W leaves W^2/m^4, so the exponent is not dimensionless. Reading 223.5 m^2/W as 1/S_c gives S_c = 0.004474 W/m^2, which is dimensionally consistent.
- The alternative reading, S_c = 223.5 W/m^2, makes S flat (0.49994 at E = 0, 0.511 at 10 W/m^2) and could not produce any alerting effect.
- T20 p. 13 places half-maximal alerting at 5x10^-2 W/m^2, consistent with S_b = 0.05.
- The correction is justified as a unit fix. Whether 1/223.5 is the value the authors actually used is doubtful; see E3.

**Recommendation**: keep as documented deviation for now, but ask the owner. E3 suggests the published curves were generated with a shallower sigmoid.

---

### E1. "atan(Y/X)" means a four-quadrant arctangent (Paper error)

**Paper expressions**: P18 Table 1 Eq 17, p. 206: `phi_crit = -2.98 = atan(Y / X)`. A18 Eq 3, PDF p. 5: `phi = atan(Y / X)`.

**How we know**: -2.98 is outside (-pi/2, pi/2). A18 states the phase wraps from -pi to +pi and plots it on a -pi..pi axis (Fig 2b, PDF p. 28). A18's phi_off = 2.78 rad also exceeds pi/2. The authors evidently mean atan2(Y, X), the common MATLAB idiom.

**Code impact**: D1.

**Recommendation**: record as a known paper notation error in the code docstring when D1 is fixed.

---

### E2. T20 S_c printed with inverted units (Paper error)

**Paper**: T20 p. 7 prints S_c = 223.5 m^2/W; Eq 14 divides (S_b - E_e,mel) by it.

**How we know**: units, as in D3. The same sentence prints S_b's unit twice, which suggests the line was not proofread.

**Code impact**: D3.

---

### E3. T20 Fig 6 does not match Eq 14 with the printed S_c under either reading (Paper error, resolution unclear)

**Paper**

- T20 Fig 6, p. 12: Delta KSS versus melanopic irradiance (log axis, 10^-4 to 10 W/m^2), comparing the default melanopic model curve with the photopic model curve and data.
- T20 p. 13: half-maximal effect at 5x10^-2 W/m^2 and saturation at 0.86 W/m^2 for the default parameters.

**Evidence**: the orange default curve was digitized from a 600 dpi render of p. 12. Axis ticks were located from pixels, so the values are approximate:

| E (W/m^2) | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 | 0.06 | 0.08 | 0.10 | 0.15 | 0.5 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Delta KSS | -0.14 | -0.30 | -0.50 | -0.70 | -0.91 | -1.11 | -1.41 | -1.61 | -1.81 | -2.01 | -2.18 |

- The direct-alerting term is linear in S(E). theta_L obeys a linear ODE driven by nu_LA * S(E), so for a constant-light protocol the E-dependence of its contribution is exactly S(E). The only other contribution is the circadian effect, which is the dashed photopic-model curve; it stays above about -0.4 for E <= 0.1.
- With S_c = 1/223.5, S(0.03) = 0.011 and S(0.10) = 0.99999: a near step between 0.04 and 0.06 W/m^2. The figure instead reaches 23% of its final value by 0.03 and 74% by 0.1.
- Subtracting a by-eye reading of the photopic curve gives a logistic width of roughly 0.015 to 0.025 W/m^2. That is about 4 to 5 times 1/223.5, and nowhere near 223.5.
- The repo's own refit agrees: `examples/sigmoid_fit.ipynb` cell 5 fits `d / (1 + exp((b - x) / c))` to the Hommes & Gimenez 2015 data converted to W/m^2. That refit produced the `S_C = 0.01604672` briefly committed in ed3e0e1.
- The stated saturation at 0.86 W/m^2 fits neither S_c = 0.0045 (S is 0.99 by 0.07) nor 0.02 (0.99 by 0.14). It probably describes the total curve including the circadian tail. Flagged, not resolved.
- A shallower S_c also makes S(0) non-negligible (about 0.08 for S_c = 0.02). That would give practical meaning to T20's remark (p. 7) that S is normalized to the 0..1 range. The code removed its explicit min-max normalization in 4ca72b2; the commented line remains in `model()`. With S_c = 1/223.5, S(0) = 1.4e-5 and the normalization is irrelevant.

**What would decide it**: the authors' code or an erratum. Failing that, simulate the Cajochen et al. 2000 protocol (6.5 h FL4100K light at night) with `models.py` for S_c = 1/223.5 and S_c = 0.02, and compare against Fig 6.

**Recommendation**: ask the owner. Do not change `S_C` until one of the checks above is done.

---

### E4. Minor paper typos with no code impact (Paper error)

- P18 Table 3, p. 209, KSS column: Theta_0 = -24.34 (22.43). The parenthetical set (the fit excluding Shekleton et al. 2013, per the table caption, read from the page image) has Theta_H = 2.13 and Theta_C = -1.74. With H near 13, a positive Theta_0 gives KSS near 50, so the sign is almost certainly dropped. The code uses the non-parenthetical defaults, which are consistent.
- T20 p. 8: the tau_L fitting range is printed as (0.24) minutes, evidently (0, 24). Note that the default tau_L = 24 min (Table 3) sits on that upper bound.

---

### U1. tau_A: 3.11 h (A18) in use, 1.5 h (T20) defined but unused (Unclear)

**Code**:

```python
TAU_A = 1.5 * 3600.0  # s  # 1.5 hours - Tekieh et al. 2020
TAU_ALPHA = 3.11 * 3600 # s
```

ODE in `model()`:

```python
(m_phi * r - A) / constants.TAU_ALPHA, # Equation 1 in [Abeysuriya et al. 2018]
```

`TAU_A` is referenced nowhere in `models.py` or `examples/sigmoid.py`.

**Papers**

- A18 Eq 1, PDF p. 4: `tau_A dA/dt = m(phi) r(I) - A(t)`. A18 Table 1, PDF p. 25: tau_A (synthesis time constant) = 3.11 x 3600 s.
- T20 Eq 8, p. 5: `tau_A dA/dt = m(phi) r(E_e,mel) - A(t)`. T20 p. 7: the text notes that maximum suppression depends on tau_A and sets tau_A = 1.5 h as the best fit to the data, in the paragraph that recalibrates the suppression function (Eq 9).
- Both papers use the symbol tau_A. No paper uses "tau_alpha"; the code name `TAU_ALPHA` is a misnomer, and alpha is the photoreceptor rate.

**Commits**: `TAU_A` came in with 8fa458a (2021-02-20), comment "Tekieh et al. 2020 - Section 2.3.2, after Equation 9". `TAU_ALPHA` came in with 76de415 (2025-05-02), without a source comment. The ODE comment was added in acd6931. No message explains the choice.

**Evidence**: both values are correctly transcribed from their papers. The code otherwise follows T20 for the melatonin block (melanopic input, T20's r_a, r_b, r_c), and T20 ties 1.5 h to that suppression function. Scratch run with `arctan2` and the T20 Hill form (one light scenario only): peak rho_b was 283.3 pmol/L at 3.11 h and 319.6 pmol/L at 1.5 h, against rho_b* = 325 pmol/L.

**Recommendation**: ask the owner. If D2 is fixed by restoring T20 Eq 9, consistency argues for tau_A = 1.5 h (`TAU_A`), and `TAU_ALPHA` should be renamed or removed. If the owner wants A18's photopic melatonin block, keep 3.11 h but then r must use A18's parameters too.

---

### U2. V_WE sign: P18 says -0.07 mV, P16 says 0.07 mV (Unclear)

**Code**: `V_WE = -0.07  # mV`, used in `wake_effort`:

```python
W = forced * max(0, constants.V_WE - constants.NU_MV * Q_v - constants.D_M)
```

**Papers**

- P18 Table 1, p. 206: V_WE = -0.07 mV.
- P16 p. 506: the best overall fit across studies is V_WE = 0.07 mV.
- P16 p. 507: repeats V_WE = 0.07 mV and adopts it as the new default. The same page prints the fitted range as -0.23 mV to 0.34 mV with the minus sign intact, so the 0.07 is not a rendering artifact.
- P18 cites P16 for further detail.

**Commit**: 8fa458a (2021-02-20), comment "wake effort constants: Postnova et al. 2018 - Table 1".

**Evidence**: the two papers by the same group disagree on the sign, and the code follows the one it cites. Impact is small: V_WE is the floor for V_m during forced wake, and Q_m at that floor is 3.37 s^-1 (V = -0.07) versus 3.52 s^-1 (V = +0.07). This changes homeostatic build-up during forced wake by about 5%.

**What would decide it**: the P16 supplementary parameter table (not available locally, not read) or the authors' code.

**Recommendation**: ask the owner. Keep -0.07 meanwhile, with a comment noting the P16 value.

---

### U3. Sleep-state gating of light for r(E) and S(E) (Unclear)

**Code**: `photoreceptor_conversion_rate` does `IE = IE * S` (P18 Eq 14, `I = IS`). `model()` calls `melatonin_suppression(IE)` and `sigmoid(IE)` with the ungated `IE`.

**Papers**

- P18 Table 1 Eq 14, p. 206: `I = IS`, defined for the photic drive.
- A18 Eq 4 uses r(I) with I defined as photopic illuminance, and does not say whether that I is gated.
- T20 Eqs 9 and 14 use E_e,mel without stating gating.
- All three papers simulate 0 lux during sleep (P16 p. 500 ambient profile; A18 PDF p. 10, read from the text extraction only), so the question never arises in their protocols.

**Evidence**: there is no difference when the input is 0 during sleep. Scenarios with light during sleep (night lights, sleeping through a lit period) will suppress melatonin and drive theta_L while the eyes are closed.

**Recommendation**: ask the owner whether light during sleep should reach r and S. Decide explicitly, not by accident.

---

## 3. Constants

Verdict key: **Match** = value and unit agree with the source. A flag points to a section above.

| Name | Code value | Paper value | Source | Verdict |
|---|---|---|---|---|
| CONVERSION_FACTOR | 754.0340838 | K_mel^D65 = 1.3262 mW/m^2/lx, so 1/K = 754.03408 lx per W/m^2 | T20 p. 4 | Match (differs at 2e-9 relative). Unused in models.py |
| TAU_V | 50.0 s | tau_v = tau_m = 50 s | P18 Table 1, p. 206 | Match |
| TAU_M | TAU_V | tau_v = tau_m = 50 s | P18 Table 1, p. 206 | Match |
| TAU_H | 59 x 3600 s | 59 x 3600 s | P18 Table 1, p. 206 | Match |
| TAU_X | 24 x 3600 / (2 pi) s | 24 x 3600 / 2pi s | P18 Table 1, p. 206 | Match |
| TAU_Y | TAU_X | tau_X = tau_Y | P18 Table 1, p. 206 | Match |
| TAU_C | 24.2 x 3600 s | 24.2 x 3600 s | P18 Table 1, p. 206 | Match |
| TAU_A | 1.5 x 3600 s | tau_A = 1.5 hours | T20 p. 7 | Value matches T20; unused. See U1 |
| TAU_L | 24 x 60 s | tau_L = 24 min (default, "All" row) | T20 Table 3, p. 6 | Match |
| TAU_ALPHA | 3.11 x 3600 s | tau_A = 3.11 x 3600 s | A18 Table 1, PDF p. 25 | Value matches A18; conflicts with T20. See U1 |
| NU_VM | -2.1 mV | -2.1 mV | P18 Table 1, p. 206 | Match |
| NU_MV | -1.8 mV | -1.8 mV | P18 Table 1, p. 206 | Match |
| NU_HM | 4.57 s | 4.57 s | P18 Table 1, p. 206 | Match |
| NU_XP | 37 x 60 s | 37 x 60 s | P18 Table 1, p. 206 | Match |
| NU_XN | 0.032 | 0.032 | P18 Table 1, p. 206 | Match |
| NU_YY | (1/3) x 37 x 60 s | 1/3 nu_Xp s | P18 Table 1, p. 206 | Match |
| NU_YX | 0.55 x 37 x 60 s | 0.55 nu_Xp s | P18 Table 1, p. 206 | Match |
| NU_VH | 1.0 | 1 | P18 Table 1, p. 206 | Match |
| NU_VC | -0.5 mV | -0.5 mV | P18 Table 1, p. 206 | Match |
| NU_LA | -0.11 | -0.11 (default, "All" row) | T20 Table 3, p. 6 | Match |
| GAMMA | 0.13 | 0.13 | P18 Table 1, p. 206 | Match |
| DELTA | 24 x 3600 / 0.99729 s | 24 x 3600 / 0.99729 s | P18 Table 1, p. 206 | Match |
| BETA | 0.007 / 60 s^-1 | 0.007 / 60 s^-1 | P18 Table 1, p. 206 | Match |
| D_M | 1.3 mV | D_m = A_m = 1.3 mV | P18 Table 1, p. 206 | Match |
| A_V | -10.3 mV | -10.3 mV | P18 Table 1, p. 206 | Match |
| V_WE | -0.07 mV | -0.07 mV (P18); 0.07 mV (P16) | P18 Table 1, p. 206; P16 p. 507 | Matches P18; papers conflict. See U2 |
| V_TH | -2.0 mV | -2 mV | P18 Table 1, p. 206 | Match |
| Q_MAX | 100 s^-1 | 100 s^-1 | P18 Table 1, p. 206 | Match |
| THETA | 10 mV | 10 mV | P18 Table 1, p. 206 | Match |
| SIGMA_PRIME | 3 mV | 3 mV | P18 Table 1, p. 206 | Match |
| EPSILON | 0.4 | 0.4 | P18 Table 1, p. 206 | Match |
| I_0 | 100 lx | 100 lx | P18 Table 1, p. 206 | Match |
| I_1 | 9500 lx | 9500 lx | P18 Table 1, p. 206 | Match |
| ALPHA_0 | 0.1 / 60 s^-1 | 0.1 / 60 s^-1 | P18 Table 1, p. 206 | Match |
| F_4100K | 8.19e-4 | 8.19 x 10^-4 (W/m^2 per lx, implied by Eq 5) | T20 Eq 5, p. 4 | Match (code comment lacks unit) |
| R | 10.0 | r = 10 | P18 Table 1, p. 206 | Match |
| S_B | 0.05 W/m^2 | 0.05 W/m^2 | T20 p. 7 | Match |
| S_C | 1/223.5 = 0.0044743 W/m^2 | 223.5 m^2/W | T20 p. 7 | Documented deviation, see D3, E2, E3 |
| R_A | 1 | r_a = 1 (T20); 0.94 (A18) | T20 p. 7; A18 Table 1, PDF p. 25 | Value matches T20; used in A18's form. See D2 |
| R_B | 0.031 W/m^2 | r_b = 0.031 W/m^2 (T20); 110 lx (A18) | T20 p. 7; A18 Table 1, PDF p. 25 | Value matches T20; used in A18's form. See D2 |
| R_C | 0.82 | r_c = 0.82, a dimensionless exponent (T20); 26 lx width (A18) | T20 p. 7; A18 Table 1, PDF p. 25 | Value matches T20; used as a width. See D2 |
| PHI_ON | -1.44 rad | -1.44 rad | A18 Table 1, PDF p. 25 | Match |
| PHI_OFF | 2.78 rad | 2.78 rad | A18 Table 1, PDF p. 25 | Match |
| U_STAR | 0.47 pmol/L/s | 0.47 pmol L^-1 s^-1 | A18 Table 1, PDF p. 25 | Match |
| A_0 | U_STAR | A_0 = u* | A18 Table 1, PDF p. 25 | Match. Unused |
| R_G | 0.9 | 0.9 | A18 Table 1, PDF p. 25 | Match |
| RHO_B_STAR | 325 pmol/L | 325 pmol L^-1 | A18 Table 1, PDF p. 25 | Match |
| T_U | 0.96 x 3600 s | 0.96 x 3600 s | A18 Table 1, PDF p. 25 | Match |
| THETA_0 | -24.34 | -24.34 (KSS) | P18 Table 3, p. 209 | Match |
| THETA_H | 2.28 | 2.28 (KSS) | P18 Table 3, p. 209 | Match |
| THETA_C | -1.74 | -1.74 (KSS) | P18 Table 3, p. 209 | Match |

Inline numeric constants in `models.py` also match:

- `circadian_drive`: 0.1, 3.1, 2.5, 4.2, 3.7 and the shift X + 2 (P18 Eq 10; P16 Eqs 1-3, p. 500).
- dX/dt polynomial: 1/3, 4/3, 256/105 (P18 Eq 4).
- `nonphotic_drive`: 2/3 (P18 Eq 11).

---

## 4. Functions and equations

| Function / ODE row | Paper equation (transcribed) | Source | Verdict |
|---|---|---|---|
| `wake_effort` | W = F_W max{0, V_WE - nu_mv Q_v - D_m} | P18 Table 1 Eq 8, p. 206 | Match |
| `total_sleep_drive` | D_v = nu_vH H + nu_vC C + A_v | P18 Eq 9 | Match |
| `nonphotic_drive` | D_n = (S - 2/3)[1 - tanh(r X)] | P18 Eq 11 | Match |
| `photoreceptor_conversion_rate`, "2018" | alpha = alpha_0 I/(I + I_1) sqrt(I/I_0), with I = I S | P18 Eqs 13, 14 | Match |
| `photoreceptor_conversion_rate`, "2020" | alpha = alpha_0 E/(E + I_1 F_4100K) (E/(I_0 F_4100K))^(1/2) | T20 Eq 7, p. 5 | Match. Code also gates E by S (P18 Eq 14). Numerically identical to "2018" for FL4100K: alpha = 0.008547 s^-1 at 10 000 lx and at 8.19 W/m^2 |
| `photic_drive` | D_p = alpha (1 - P)(1 - eps X)(1 - eps Y) | P18 Eq 12 | Match |
| `mean_population_firing_rate` | Q_i = Q_max / (1 + exp([theta - V_i]/sigma')) | P18 Eq 7 | Match |
| `state` | S = 1 if V_m > V_th (wake), 0 if V_m <= V_th | P18 Eq 15 | Match |
| `sigmoid` | S(E) = 1 / (1 + exp((S_b - E)/S_c)) | T20 Eq 14, p. 7 | Form matches; S_C flagged (D3, E3) |
| `alertness_measure` | AM = theta_C C + (theta_H + theta_L) H + theta_0 | T20 Eq 12, p. 7; P18 Eq 23, p. 209 | Match |
| `circadian_drive` | C = 0.1 (1 + X)/2 + [(3.1X - 2.5Y + 4.2)/(3.7(X + 2))]^2 | P18 Eq 10; P16 Eqs 1-3, p. 500 | Match |
| `melatonin_suppression` | r(E) = 1 - r_a / (1 + (E/r_b)^(-r_c)) | T20 Eq 9, p. 7 | **Code bug D2** (A18 Eq 4 logistic form with T20 parameters; docstring mis-cites A18) |
| `circadian_phase` | phi = atan(Y/X), meaning four-quadrant | A18 Eq 3, PDF p. 5; P18 Eq 17 | **Code bug D1** |
| `melatonin_synthesis_regulation` | m = 1 if phi < phi_on or phi > phi_off; 0 if phi_on <= phi <= phi_off | A18 Eq 2, PDF p. 5 | Match |
| `urinary_excretion_rate` | u(t) = u* rho_b(t - t_u) / rho_b* | A18 Eq 15, PDF p. 9 | Match. The delayed value is a previous-sample lookup; before t = 0 it uses the first sample |
| `forced_wake` | F_W = 1 if forced wake and 0 otherwise | P18 Table 1, p. 206 | Match (definition). The 06:00 to 22:00 daily schedule, inclusive at both ends, is a scenario choice not in the papers |
| `irradiance` | none | none | No paper equation. Clamps interpolated light at 0 (added in 4ca72b2) |
| `model` dV_v/dt | tau_v dV_v/dt = nu_vm Q_m - V_v + D_v | P18 Eq 1 | Match |
| `model` dV_m/dt | tau_m dV_m/dt = nu_mv Q_v - V_m + D_m + W | P18 Eq 2 | Match |
| `model` dH/dt | tau_H dH/dt = nu_Hm Q_m - H | P18 Eq 3 | Match |
| `model` dX/dt | tau_X dX/dt = Y + gamma(X/3 + 4X^3/3 - 256 X^7/105) + nu_Xp D_p + nu_Xn D_n | P18 Eq 4 | Match |
| `model` dY/dt | tau_Y dY/dt = D_p(nu_YY Y - nu_YX X) - (delta/tau_c)^2 X | P18 Eq 5 | Match |
| `model` dP/dt | dP/dt = alpha(1 - P) - beta P | P18 Eq 6 | Match |
| `model` dtheta_L/dt | tau_L dtheta_L/dt = -theta_L + nu_LA S(E) | T20 Eq 13, p. 7 | Match |
| `model` dA/dt | tau_A dA/dt = m(phi) r - A | A18 Eq 1, PDF p. 4; T20 Eq 8, p. 5 | Form matches. tau_A flagged (U1); inputs m and r carry D1 and D2 |
| `model` drho_b/dt | drho_b/dt = u* A / r_g - u* rho_b / (r_g rho_b*) | A18 Eq 14, PDF p. 9 | Match |
| `model_run` y0 | none | none | Not in any paper. `[-4.55, -0.07, 13.29, -0.14, -1.07, 0.10, ...]` came from run.ipynb in 8395c8d, commented "proper initial values after experimentation". A = 0 and rho_b = 25 were added in 76de415; theta_L was set to 0 in 4ca72b2. At t = 0 (midnight) the model is awake (V_m = V_WE) with A = 0, so the first days are spin-up |

---

## 5. Notes outside the equation audit

- `examples/sigmoid.py` differs from `models.py` only in exposing `S_B`, `S_C`, `NU_LA`, `TAU_L` as arguments. It inherits D1, D2 and U1.
- `examples/sigmoid_fit.ipynb` cell 5 converts melanopic EDI to irradiance with `0.91 * 1.2013e-3` W/m^2 per lx. T20 p. 4 gives K_mel^D65 = 1.3262e-3, which is also what `CONVERSION_FACTOR` encodes. The same cell plots the "Original Curve" with b = 0.3, not `constants.S_B`.
- The fitted amplitude `d` in that cell is a Delta-KSS amplitude, not nu_LA: nu_LA multiplies H in Eq 12. That is why the commented `NU_LA = -1.619739` in ed3e0e1 is roughly 15 times the T20 value; the ratio is about the size of H.
- `model_run` hard-codes `version_year = "2020"`, so the "2018" branch of `photoreceptor_conversion_rate` is unreachable from `model_run`.
- `model_run` builds the light input with `interp1d(..., kind='cubic')`. On step-shaped light profiles this overshoots and undershoots. Undershoot is clamped by `irradiance()`; overshoot is not.

## 6. Not verified / not read

- P16 Table 1 and the P16/P18 supplementary material: not read (not available locally). This is what U2 needs.
- A18 Eqs 5-7 and the saliva equation: read from the text cache only. They are not used in the code.
- T20 Fig 6 values are digitized and approximate. The photopic-curve subtraction was done by eye.
- Phillips 2019: not read beyond a grep; nothing in the code cites it.
- Hommes & Gimenez 2015 data and the notebook's outputs: not examined (outputs ignored as instructed).
- The scratch simulations cover a single light scenario and are supporting evidence only.
