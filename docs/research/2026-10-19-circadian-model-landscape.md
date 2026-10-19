# Circadian, sleep and alertness models with light input: landscape for a multi-model library

Date: 2026-10-19. This document covers four things: models a library built from this repo could host, prior-art libraries, the phase bug in `models.py`, and the reference data that exists. Every claim cites a numbered source at the end. "Not read" means I could not open the full text. Those items are collected under [Could not verify](#could-not-verify).

## Summary for the interface designer

1. **Time units differ between models.** The Postnova/Tekieh model and Skeldon 2023 are written in seconds [3][18]. Kronauer, Forger, St Hilaire, Hannay and Arcascope `circadian` use hours [24][25][41]. Each model should declare its time unit, and the library should convert at the boundary.
2. **Light must be a typed quantity, not a bare float.** Most models take photopic lux [3][18][24][25]. Tekieh 2020 takes melanopic irradiance in W/m² [4], and nvRD takes ipRGC-weighted W/m² [35]. Studies that report melanopic EDI get it from lux with an assumed M:P ratio [38][40]. Inputs should carry a quantity and a unit, and the scalar mEDI-to-W/m² factor (1.3262 mW/lm) belongs in the library [47]. Spectrum handling should stay optional, because luxpy is GPLv3 [47].
3. **Sleep/wake can be absent, a scheduled input, or emergent.** It is absent in Kronauer, Forger and Hannay; a scheduled input in St Hilaire's σ and the three-process model; and emergent in Phillips-Robinson, Postnova and Skeldon [24][32][3][18]. Forced wake is a separate input [3][17]. Inputs should therefore be named time series (`light`, `forced_wake`, `sleep_schedule`), each model should declare which ones it needs, and a model with emergent sleep must gate its own light [3][18].
4. **Most of these models are hybrid and stiff.** Postnova has a V_m threshold switch, step inputs, and time constants from 50 s to 59 h [3]. Skeldon 2023 uses hysteresis switching, a stiff solver and event location [18]. The solver should be chosen per model and support events and input breakpoints. Arcascope's fixed-step RK4, which keeps discrete state in instance attributes, is the pattern to avoid [41].
5. **Each model owns its phase markers.** CBTmin is variously min x + 0.8 h, min x + 0.97 h, ψ = π, or a four-quadrant angle rule (-2.98 rad + 2.7 h in Postnova 2018) [3][22][24][25]. DLMO is t(φ = -1.44) + 0.5 h, a 20 pmol/L threshold, or a code convention of CBTmin - 7 h [5][41]. The library should share helpers (atan2, crossing and peak finding), not one definition.
6. **Derived outputs need the whole trajectory plus the inputs.** KSS uses the states and light through Θ_L [4], and urinary aMT6s is a delayed function of plasma melatonin, u(t) ∝ ρ_b(t - T_U) [2][58]. These belong in post-processors over a dense trajectory, not in extra ODE states.
7. **Parameter sets need version, citation and units.** This repo already carries a 2018 and a 2020 variant [3][4]. The group's photic-drive constants differ between 2013 and 2018 [6][3], and two Forger 1999 parameter sets circulate [25][17].
8. **There are no public reference trajectories.** The group's code is "available upon request" [5]. The only open implementation, NIF_Photobiology [58], publishes aggregates only. Tests must rest on invariants and published summary numbers [3][5][7].

## Comparison table

"Reference code" lists openly readable implementations only. Lux means photopic lux unless stated otherwise.

| Model | Citation | States | Light input, units | Sleep/wake | Outputs | Reference code |
|---|---|---|---|---|---|---|
| Arousal dynamics, photopic | Postnova 2016/2018 [1][3]; melatonin from Abeysuriya 2018 [2] | V_v, V_m, H, X, Y, P; melatonin adds A, ρ_b | lux, α = α0 · I/(I + I1) · √(I/I0), I0 = 100 lx, I1 = 9500 lx; gated I·S (Eq 13-14) [3] | Emergent (MA/VLPO flip-flop); forced-wake input F_W [3] | Sleep state, H, C, CBTmin and MELpeak (Eq 16-17), KSS, PVT, DSST [3]; DLMO, melatonin, aMT6s [2][5] | NIF_Photobiology (Python, GPL-3.0) [58]; group code "on request" [5]; closed Alertness API [59] |
| Arousal dynamics, melanopic | Tekieh 2020 [4] | Adds Θ_L | Melanopic irradiance, W/m² (Eq 1, 7) [4] | As above | As above, plus light-dependent KSS weight (Eq 12-14) [4] | NIF_Photobiology [58] |
| Sleep-wake switch | Phillips & Robinson 2007 [11] | V_v, V_m, H | None (prescribed C) [15][20] | Emergent | Sleep state, H | Stan port, CellML, R [55] |
| Switch + light oscillator (PCR) | Phillips, Chen, Robinson 2010 [12]; restated in [14][17] | V_v, V_m, H, x, y, n | lux via Forger 1999 Process L, gated by Q_m threshold [17] | Emergent | Sleep timing, CBTmin | None found |
| Process L + higher-order VdP | Kronauer, Forger, Jewett 1999 [21]; Jewett, Forger, Kronauer 1999 [22] | x, xc, n | lux, α = α0(I/I0)^p [24] | None | CBTmin = min x + 0.8 h [22] | Arcascope `Jewett99` [41] |
| Simpler cubic VdP | Forger, Jewett, Kronauer 1999 [23] | x, xc, n | lux, Process L [25] | None | CBTmin = min x [25] | Arcascope `Forger99` [41]; hcrsimpy [42] |
| Photic + non-photic | St Hilaire 2007 [24] | x, xc, n | lux, α = α0(I/I0)^p · I/(I+100) (Eq 9) [24] | Input: scheduled σ ∈ {0,1} (Eq 10) [24] | CBTmin where arctan(xc/x) = -170.7°, + 0.97 h (Eq 7-8) [24] | Arcascope `Hilaire07` (deviates) [41]; predicting_dlmo JS [44] |
| Macroscopic, single population | Hannay, Booth, Forger 2019 [25] | R, ψ, n | lux (inferred from fits) | None | CBTmin at ψ = π [25] | Arcascope `Hannay19` [41]; hcrsimpy [42] |
| Macroscopic, two population | Hannay, Booth, Forger 2019 [25] | R_v, R_d, ψ_v, ψ_d, n | lux, ventral only | None | CBTmin at ψ_v = π [25] | Arcascope `Hannay19TP` [41] |
| Modified PCR | Skeldon, Phillips, Dijk 2017 [17] | V_v, V_m, H, x, y, n | lux, gated | Emergent; alarm as wake effort | Sleep timing, social jetlag | None released [17] |
| HCL | Skeldon et al. 2023 [18] | H, x, y, n + binary S | lux, Ĩ = (1-S)I (Eq 5) [18] | Emergent (hysteresis on H) | Sleep timing, CBTmin, DLMO | MATLAB, Surrey non-commercial licence [53]; Arcascope `Skeldon23` [41] |
| Two-process | Borbély 1982 [29]; Daan, Beersma, Borbély 1984 [30]; restated in [19] | S | None | Emergent from thresholds, or imposed | Sleep timing, S | MATLAB app (MIT) [54] |
| Three-process | Åkerstedt, Folkard, Portin 2004 [31]; equations in [32] | S, C, U, W (closed form) | None | Input (sleep/duty times); optional sleep generator | Alertness; KSS = 10.6 - 0.6(S+C+U+W) [32] | FIPS (R, AGPL-3) [45] |
| Alertness and throughput | Jewett & Kronauer 1999 [33] | H, C (pacemaker), W | lux via pacemaker | Not read | Subjective alertness, throughput | None found |
| Exogenous melatonin | Breslow 2013 [28] | x, xc, n, H1-H3 (per [41]) | lux | None | Melatonin, CBTmin | Arcascope `Breslow13` (untested) [41] |
| nvRD direct response | Amundadottir et al. 2017 [35] | Filter chain, no ODE | ipRGC-weighted W/m² [35]; mEDI in LightLogR [46] | None | Relative response r_D, not KSS | LightLogR `nvRD` (R, MIT) [46] |
| Static ΔKSS fit | Hommes & Giménez 2015 [34] | None | Melanopic illuminance | n/a | ΔKSS | Refit in this repo's `examples/sigmoid_fit.ipynb` |
| Adenosine / PVT | Phillips, Klerman, Butler 2017 [36] | Adenosine, receptor states | None | Input | PVT lapses | MATLAB, no licence [36] |
| Stochastic PVT | Raison et al. 2025 [8] | SDE | None | n/a | PVT reaction-time distribution | Python, no licence [8] |

## Models

### Postnova/Robinson arousal dynamics (this repo)

**Lineage**
- Phillips & Robinson 2007: the MA/VLPO mutual-inhibition switch with homeostat H. The circadian drive was prescribed, and there was no light [11][15][20].
- Phillips, Chen & Robinson 2010: the first coupling to a light-driven oscillator, Forger 1999 [12][13].
- Phillips 2011 and Postnova 2013: switched to the St Hilaire 2007 form with a non-photic drive [13][6]. Postnova 2016 builds on that form.
- Postnova 2016: revised the circadian drive to C = 0.1(1+X)/2 + [(3.1X - 2.5Y + 4.2)/(3.7(X+2))]² to produce the wake maintenance zone [1][3].

**Postnova 2018 Table 1** [3]
- I read Table 1 from an archived copy of the publisher page. It lists Eq 1-17 and all parameters.
- Time constants: τ_v = τ_m = 50 s, τ_H = 59 × 3600 s, τ_X = τ_Y = 24 × 3600/2π s.
- Wake effort: W = F_W max{0, V_WE - ν_mv Q_v - D_m} (Eq 8). The term goes back to Fulcher, Phillips & Robinson 2010, where wake effort is the correlate of subjective fatigue [37] (abstract only).
- Non-photic drive: D_n = (S - 2/3)[1 - tanh(rX)] (Eq 11). This equals St Hilaire's ρ(1/3 - σ) with S = 1 - σ [24].
- Photic rate: α = α0 · I/(I + I1) · √(I/I0), with I0 = 100 lx and I1 = 9500 lx (Eq 13). Light is gated as I = I·S (Eq 14).
- State function: S = 1 if V_m > V_th, with V_th = -2 mV (Eq 15).
- Phase markers: t_crit = t_φcrit + t0, with φ_crit = -2.98 = atan(Y/X). t0 is 0.7 h for MELpeak and 2.7 h for CBTmin (Eq 16-17).
- KSS (Table 3): KSS = Θ0 + Θ_H H + Θ_C C, with Θ0 = -24.34, Θ_H = 2.28, Θ_C = -1.74. Table 3 also gives PVT and DSST weights.
- Default entrainment: sleep 23:30 to 08:00, with 250 lux from 08:00 to 20:00, 40 lux otherwise when awake, and 0 lux during sleep.
- Postnova 2013 wrote the photic rate as α0(I/I0)^0.5 · I/(I + I1) with I0 = 9500 lux and I1 = 100 lux [6]. That is a different function, about 5× larger at 100 lux. This repo follows 2018.

**Tekieh 2020** [4] (accepted manuscript, read via an archived copy; may differ from the version of record)
- Melanopic irradiance replaces lux (Eq 1). The photic rate becomes α = α0 · E/(E + I1 F_4100K) · √(E/(I0 F_4100K)), with F_4100K = 8.19 × 10⁻⁴ (Eq 5, 7).
- Melatonin: τ_A dA/dt = m(φ) r(E) - A, with τ_A = 1.5 h (Eq 8).
- Suppression: r(E) = 1 - r_a/(1 + (E/r_b)^(-r_c)), with r_a = 1, r_b = 0.031 W/m² and r_c = 0.82 (Eq 9).
- KSS: KSS = θ_C C + (θ_H + Θ_L) H + θ0 (Eq 12), with τ_L dΘ_L/dt = -Θ_L + ν_LA S(E) (Eq 13) and S(E) = 1/(1 + exp((S_b - E)/S_c)) (Eq 14).
- Parameter values: ν_LA = -0.11, τ_L = 24 min, S_b = 0.05 W/m², and S_c printed as 223.5 m²/W. This repo's `S_C = 1/223.5` ("typo corrected from paper") and NIF's identical value, set "as per Dr. Tekieh's and Dr. Postnova's emails", both invert the printed value [58].
- Studies that used a pupil dilator get E × 10.
- The fitting used MATLAB `patternsearch`.

**Inputs and outputs**
- Inputs: light (lux or W/m²) and forced wake. In Knock 2021, a self-reported sleep diary is converted into forced-wake windows [7].
- Outputs: sleep state, KSS, PVT, DSST, CBTmin/MELpeak, DLMO, plasma melatonin and aMT6s [2][3][5][7].
- In Knock 2021, aMT6s acrophase comes from a cosine fit to 2 h means [7].

**Integration**
- The system is stiff, with time constants from 50 s to 59 h, and discontinuous at the V_m threshold and at forced-wake and light steps.
- No group paper I read names its solver [3][4][5][7].
- NIF_Photobiology integrates with forward Euler at a default 20 s step [58].

**Code**
- The group's code is not public. Huang 2026 says it is "available upon request on GitHub" [5].
- Knock 2021 points users to the Alertness API [7][59].
- NIF_Photobiology (Python, GPL-3.0) is an independent implementation of Postnova 2018, Abeysuriya 2018 and Tekieh 2020. Its changelog records corrections "per Dr. Postnova and Dr. Pierson" [58].

**Successors 2021 to 2026**
- These papers apply the model without adding states: Knock 2021 [7], Varma 2024/2025 [9][10], and Huang 2026 [5], which restates the full model in Supplementary Table S1.
- Raison 2025 adds a separate stochastic PVT model, regressed on H and C [8].
- I found no successor with new states such as light history or a spectral input.

### Kronauer, Forger, Jewett 1999 (Process L + higher-order van der Pol)

- **Structure.** States x, xc, n, in hours (π/12 rad/h). Process L: α = α0(I/I0)^p, dn/dt = 60[α(1-n) - βn], B̂ = G(1-n)α, B = B̂(1 - 0.4x)(1 - 0.4xc). These are as restated by St Hilaire 2007 Eqs 1-6, which give prior values α0 = 0.05, p = 0.5 and I0 = 9500 [24].
- **CBTmin.** 0.8 h after the minimum of x [22] (abstract).
- **Inputs.** Light is the only input.
- **DLMO.** DLMO = CBTmin - 7 h is a convention in later code [27][41], not in the 1999 papers.
- **Parameters.** Arcascope `Jewett99` uses G = 19.875, α0 = 0.16, p = 0.6 and β = 0.013, which disagree with St Hilaire's quoted prior values. I could not resolve this because the 1999 papers and the 2000 erratum were not read.

### Forger, Jewett, Kronauer 1999 (simpler model)

- **Structure.** Cubic van der Pol: dx/dt = (π/12)(xc + B) and dxc/dt = (π/12){μ(xc - 4xc³/3) - x[(24/(0.99669 τx))² + kB]}, plus Process L. CBTmin = min x [25] (Eq 1-2). Light is the only input. Units are hours.
- **Parameters.** Two sets circulate.
  - Serkh & Forger 2014, used by Hannay 2019: G = 33.75, α0 = 0.05, β = 0.0075, p = 0.5 [25][26].
  - The PCR, Skeldon and HCL lineage: G = 19.9, α0 = 0.16, β = 0.013, p = 0.6 [17][18].
- **Code.** Arcascope `Forger99` (RK4) [41]; hcrsimpy (`solve_ivp(..., method='Radau')`) [42].

### St Hilaire et al. 2007

- **Model** [24]:
  - α = α0(I/I0)^p · I/(I+100), with α0 = 0.1 min⁻¹, β = 0.007 min⁻¹, G = 37 (Eq 9).
  - Non-photic drive N̂_s = ρ(1/3 - σ), where σ = 1 for sleep/rest and 0 for wake, and ρ = 0.032 (Eq 10). It is modulated as N_s = N̂_s(1 - tanh 10x) (Eq 12).
  - It models "scheduled sleep and wake, rather than actual sleep and wake".
  - CBTmin: the time when arctan(xc/x) = -170.7°, plus φ_ref = 0.97 h (Eq 7-8). -170.7° is a four-quadrant angle.
- **Integration.** The paper states no solver. The right-hand side is discontinuous at σ steps, at the Eq 14 window and at light steps.
- **Arcascope `Hilaire07` deviates from the paper** [41]:
  - It derives the non-photic window from a fixed clock time, computed by adding -2.98 rad to 0.97 h.
  - It takes CBTmin as min x + 0.97 h instead of using the angle rule.
  - Its input checks validate rows instead of columns.

### Hannay, Booth, Forger 2019

- **Single-population model** [25]:
  - States R (amplitude), ψ (mean phase) and n.
  - Light enters through α(L) = α0 L^p/(L^p + I0), with no sensitivity modulator.
  - CBTmin is at ψ = π.
- **Two-population model** [25]: ventral and dorsal (R, ψ) pairs. Light reaches the ventral population only, and CBTmin is at ψ_v = π.
- **Units and solver.** Units are hours. The paper used custom C++ with a variable-step explicit RK4 [25].
- **Singularity.** The ψ equation contains 1/R, which is singular as R → 0 under strong light. This is my observation, not the authors'.
- **Parameters.** Code values (p = 1.5, I0 = 9325, τ = 23.84 h) agree between hcrsimpy and Arcascope [41][42]. The paper's supplement was not read.

### Phillips-Robinson models with light (PCR 2010, Skeldon 2017)

- **Light.** PCR couples the switch to Forger 1999 Process L in lux. Light is gated when Q_m falls below threshold. Parameters per [17] Eqs 1-10 and [14] Table 1: I0 = 9500 lux, p = 0.6, α0 = 0.16, β = 0.013, G = 19.9, b = 0.4, k = 0.55, γ = 0.23, f = 0.99669.
- **Skeldon 2017 changes** [17]:
  - Circadian drive changed to C = ½(1 + 0.80y - 0.47x).
  - An alarm is modelled by no action, an instant state reset, or a raised D_m ("wake effort"), depending on regime (supplement Eq 13).
  - Light scale: 70 lux per W/m².
  - Integration with MATLAB `ode15s`.
- **Eyelid gating varies.** Most papers zero light during sleep. Swaminathan 2017 passes 3% [14].

### Skeldon et al. 2023 (HCL)

- **Equations** [18]:
  - χ dH/dt = -H + (1-S)μ.
  - Sleep starts at H⁺ = H̄0 + Δ/2 + c_a C and ends at H⁻ = H̄0 - Δ/2 + c_a C.
  - Light is gated: Ĩ = (1-S)I.
  - Input is Actiwatch lux at 1 min resolution.
- **Units.** Seconds: χ = 45 × 3600 s and α0 = 0.16/60 s⁻¹.
- **Solver.** MATLAB `ode15s` with RelTol = AbsTol = 1e-4, and "the event solver was used to locate switching points".
- **Code.** [53], under a University of Surrey licence (non-commercial research only, no redistribution).
- **Arcascope port.** `Skeldon23` uses μ = 17.78. Table 1 of the paper gives 17.87 [18][41].

### Borbély two-process model

- **Formulation.** The originals were not read [29][30]; this is the restatement in Skeldon & Dijk 2025 [19]. S relaxes toward upper and lower asymptotes, with χ_w = 18.2 h and χ_s = 4.2 h. Switching happens at H⁺ = H0⁺ + aC(t) and H⁻ = H0⁻ + aC(t), with C = cos ωt.
- **Light and integration.** No light input. The solution is piecewise exponential with event switching.
- **Why it matters.** It supplies the S/C vocabulary used by the three-process KSS mapping and by Postnova's weighted-drive KSS [3][32].

### Åkerstedt/Folkard three-process model

- **Source.** The 2004 paper was read as an abstract only [31]. The equations come from Ingre et al. 2014 [32].
- **S (homeostatic).** Rises during wake (ha = 14.3, la = 2.4, d = -0.0353) and recovers in sleep, with a brake at 12.2.
- **C (circadian).** C = 2.5 cos(2π(t - 16.8)/24).
- **U (ultradian).** U = -0.5 + 0.5 cos(2π(t - 19.8)/12).
- **W (sleep inertia).** W = -5.72 exp(-1.51 t_aw).
- **KSS.** KSS = 10.6 - 0.6(S + C + U + W).
- **Inputs and solution.** Sleep and duty times, in hours. No light. The model is closed form.
- **FIPS** implements it alongside the "unified" model and has a stored ground-truth regression test [45].

### Other light-to-alertness models

- **Jewett & Kronauer 1999** [33]: H, the pacemaker output and sleep inertia drive alertness and cognitive throughput. Abstract only.
- **Hommes & Giménez 2015** [34]: a static sigmoid of ΔKSS against melanopic illuminance. It is the shape behind Tekieh's S(E) and this repo's `sigmoid_fit.ipynb`.
- **Amundadottir nvRD** [35]:
  - The input is ipRGC-weighted irradiance in W/m².
  - It applies a boxcar, an adaptive Hill function, a light-history term and an exponential filter at Δt = 0.1 h, and gives a relative response, not KSS.
  - LightLogR implements it with mEDI input [46].
  - It is the only dynamic model besides Tekieh 2020 that is native to a melanopic quantity.
- **Melanopic EDI in practice:** studies using Phillips-Robinson or Skeldon models convert lux with a fixed M:P ratio. Stone 2025 uses 0.55; Xu 2024 uses 0.63 and 0.52 [38][40]. Papatsimpa 2021 proposes substitution via mel-DER [39].

## Prior-art libraries

### Arcascope `circadian` (Python), read in depth

I read the PyPI sdist 1.0.3 (uploaded 2025-08-18) directly. GitHub `main` at db6eb08 (2025-09-12) was read by a delegated search agent [41].

**Code generation**
- Every module begins "AUTOGENERATED! DO NOT EDIT!" and names its source notebook. For example, `circadian/models.py` is generated from `nbs/api/00_models.ipynb`.
- Methods are attached to classes with fastcore `@patch_to`.

**Base class: `circadian.models.CircadianModel(ABC)`**
- `__init__(self, default_params: dict, num_states: int, num_inputs: int, default_initial_condition: np.ndarray)`.
- Each model implements `derv(t, state, input)`, `phase(trajectory=None, time=None)`, `amplitude(...)`, `cbt(trajectory=None)` and `dlmos(trajectory=None)`.
- The base class provides `step_rk4`, `integrate(time, initial_condition=None, input=None)`, `__call__`, `get_parameters_array()` and `equilibrate(time, input, num_loops=10)`.
- There is no `@abstractmethod`. The base `derv` *returns* `NotImplementedError` instead of raising it.

**Parameters**
- Parameters are a flat dict of floats. Each key is `setattr` onto the instance, and the RHS reads them as `self.taux` and so on.
- Override with `Model(params={...})` or `model.parameters = {...}`.
- The setter stores the partial dict. After `Hannay19(params={'K': 0.0, 'gamma': 0.0})`, `parameters` holds 2 keys while the attributes keep their defaults.
- Marker offsets (`cbt_to_dlmo`, `phi_ref`) and run options (`S0`, `forced_wakeup_light_threshold`) live in the same dict.
- No units, no provenance.

**Light schedules**
- `circadian.lights.LightSchedule(light, period=None)` is a vectorised callable from t (h) to lux.
- Builders: `Regular`, `ShiftWork`, `SlamShift`, `SocialJetlag`, `Hilaire12`, `Chang14`, `School`, `from_pulse`.
- Combinators: `+`, `-`, `concatenate_at`.
- The user samples a schedule onto the time grid. `integrate` then receives a numpy array of shape `(n,)`, or `(n, 2)` with light and wake columns for `Hilaire07`.
- Inputs are lux only; there is nothing melanopic.
- The wake conventions conflict: `Hilaire07.derv` treats wake = 1 as awake, while `_get_default_initial_condition` sets wake = 1 when dark.

**Integration**
- Hand-written fixed-step classical RK4, in hours. The step size is the spacing of the user's time array.
- There is no scipy solver and no event handling.
- Input is held at the step-end sample. All four RK4 stages are called with the step-end `t`.
- An initial condition of shape `(num_states, batch)` integrates a batch in one call.
- `Skeldon23` refuses batch mode, because its sleep state lives in instance attributes (`current_sleep_state`, `sleep_state`, `received_light`) that an overridden `step_rk4` mutates.

**Outputs and markers**
- `integrate` returns and caches a `DynamicalTrajectory(time, states)`. It supports `__call__(t)` (linear interpolation), indexing and `get_batch`.
- Markers are found with `scipy.signal.find_peaks`: minima of x (plus `phi_ref` for `Jewett99`/`Hilaire07`), or peaks of -cos ψ (`Hannay19`).
- Every `dlmos()` returns `cbt() - cbt_to_dlmo`, with `cbt_to_dlmo` = 7.0 h.
- `phase()` is `np.angle(x - i·xc)`, a four-quadrant angle.

**Models and modules**
- Models: `Forger99`, `Hannay19`, `Hannay19TP`, `Jewett99`, `Hilaire07`, `Breslow13`, `Skeldon23`.
- `sleep.py` contains a `TwoProcessModel` labelled "Under construction". `metrics.py` contains `esri`.
- There is no arousal model, no KSS and no endogenous melatonin output.

**Tests**
- nbdev notebook tests (`nbs/test/test_models.ipynb` and others) run on push and pull request via `fastai/workflows/nbdev-ci@master`.
- They check input validation, `derv` against hand-computed values at one state, markers against re-implementations, and convergence over 80 days on a regular schedule.
- Nothing is compared against published trajectories or PRCs. `Breslow13` is untested.

**Licence and maintenance**
- The licence is ambiguous. The `LICENSE` file is MIT (`Copyright (c) [2023] [Kevin Hannay]`), but `settings.ini` says `license = apache2`, so PyPI shows Apache 2.0.
- PKG-INFO says "Development Status :: 4 - Beta" and supports Python 3.7 to 3.10.
- Releases run from 0.0.1 (2023-01-27) through 1.0.2 (2023-08-02), then 1.0.3 (2025-08-18). The last commit is from 2025-09-12, and there are 12 open issues and PRs.
- The 1.0.3 sdist omits `sample_data/` because `package_data` names the wrong directory.

**What to borrow**
- The small per-model surface (`derv` plus marker methods).
- Batched initial conditions.
- An interpolating trajectory object.
- Composable light schedules.

**What to avoid**
- Fixed-step RK4 as the only solver.
- Unitless array inputs.
- Flat parameter dicts mutated with `setattr`.
- Discrete state kept in instance attributes.
- A hard-coded DLMO = CBTmin - 7 h.

Depending on it directly would not bring an arousal model and would inherit its licence ambiguity.

### Other multi-model libraries

| Library | Language, licence | Models | Notes |
|---|---|---|---|
| hcrsimpy [42] | Python; LICENSE says MIT, setup.py classifier says GPLv3; archived 2023 | `SinglePopModel`, `TwoPopModel`, `vdp_forger99_model` | `solve_ivp(method='Radau')`; light as a callable; `updateParameters(dict)` |
| HCRSim.jl [43] | Julia; no licence file; archived 2023 | `Forger99`, `Hilaire2007Model`, `SinglePopModel` (`<: CircadianModel`) | `RK4()` with `saveat`; callbacks for marker times |
| FIPS [45] | R; AGPL-3; last commit 2023-08-07; not on CRAN | Three-process, "unified" | `FIPS_simulate(df, modeltype, pvec)`; validated named parameter vectors; sleep-times input; regression test against a stored ground truth |
| LightLogR [46] | R; MIT; CRAN 0.10.6 (2026-09-24) | Amundadottir `nvRC`, `nvRD` | Light-logger analysis package; `MEDI.vector` input |
| predicting_dlmo [44] | JavaScript; README says MIT, no LICENSE file | FJK 1999, Jewett 1999, St Hilaire 2007 | Fixed RK4 at 1 min; DLMO = min x - 7 h |
| NIF_Photobiology [58] | Python 2.7/3; GPL-3.0 | Postnova 2018 + Abeysuriya 2018 + Tekieh 2020 | Single model, not a library. Forward Euler at 20 s; daily loop; building-simulation oriented |

Other single-model code: Phillips-Robinson ports in Stan, CellML and R [55]. A MATLAB toy and an XPPAUT course project carry the Postnova name but are not this model [56].

### Light-unit conversion (CIE S 026)

| Library | Language, licence | What it provides | Input |
|---|---|---|---|
| luxpy [47] | Python, GPLv3 | `luxpy.toolboxes.photbiochem` exports `spd_to_aopicE`, `spd_to_aopicEDI`, `spd_to_aopicDER`, `spd_to_aopicELR`, `spd_to_aopicX`. Its D65 melanopic ELR is 1.32621318911359 mW/lm. | A spectrum in W/m². For lux + CCT, first build a spectrum with `blackbody(cct)` or `daylightphase(cct)`. |
| colour-science [48] | Python, BSD-3 | No α-opic or melanopic functions in 0.4.7 | n/a |
| LightLogR [46] | R, MIT | `spectral_integration(..., action.spectrum = "melanopic", general.weight = "auto")` returns mEDI using 1/0.0013262 | A spectrum |
| pysilsub, pyplr [49][50] | Python, MIT | α-opic irradiance only, no EDI | Spectra |
| CIE S 026 α-opic Toolbox [51] | Excel | Reference tool | Pasted spectrum |
| luox [52] | JavaScript web app, GPL-3.0 | Reference tool | Spectrum |

Two consequences for the library:
- Scalar conversion between mEDI (lx) and melanopic irradiance (W/m²) needs only the D65 constant. This repo's `CONVERSION_FACTOR = 754.034` is 1/0.0013262.
- Going from photopic lux to mEDI needs a spectrum or an assumed mel-DER, and the API should force that assumption to be stated.

## Phase definition and melatonin switch: atan vs atan2

Verdict: the code does not match. Postnova 2018 Eq 17 sets "φ_crit = -2.98 = atan(Y/X)". -2.98 rad is outside the range (-π/2, π/2) of the principal arctangent, so the paper's "atan" is the four-quadrant angle, and `circadian_phase` must return `numpy.arctan2(Y, X)`. The switch convention in `melatonin_synthesis_regulation` (0 inside [PHI_ON, PHI_OFF], 1 outside) is correct, but it only works with that four-quadrant phase.

**What I read and what I could not**

- **Postnova 2018 Table 1** (READ, archived copy of the publisher page [3]). Eq 16 is t_crit = t_φcrit + t0 and Eq 17 is φ_crit = -2.98 = atan(Y/X), with t0 = 0.7 h for MELpeak and 2.7 h for CBTmin. The MathML confirms the literal `atan` and the `Y/X` fraction.
  - Eq 17 is the phase-marker definition, not the melatonin window.
  - With `numpy.atan`, Eq 17 has no solution, so the repo can compute neither CBTmin nor MELpeak.
- **Abeysuriya 2018 Eq 2** (NOT READ). Wiley serves a bot-verification page, and Europe PMC, Unpaywall and OpenAlex list no repository copy [57]. The window convention therefore rests on three other sources:
  - **Huang 2026** [5], Supplementary Table S1. It points to Postnova 2018 and Abeysuriya 2018 "for further details" and gives:
    - Eq 12: "ϕ = atan(Y/X)".
    - Eq 13: m(ϕ) = 0 for ϕ_on ≤ ϕ ≤ ϕ_off, else 1.
    - Eq 14: t_onset = t(ϕ = ϕ_on).
    - ϕ_on = -1.44 rad and ϕ_off = 2.78 rad.
    - Main text Eq 7: t_DLMO = t(ϕ_on) + 0.5 h.
  - **Tekieh 2020** [4], Eq 8. It uses m(φ) "defined and calibrated in the original model".
  - **NIF_Photobiology** [58], an independent implementation. It computes `theta = math.atan2(Y, X)` and sets m = 1 if `theta < theta_on or theta > theta_off`, which is exactly the reading above.
- **The same notation elsewhere.** Postnova 2013 writes "φ_crit = -2.98 = arctan(x/x_c)" [6], and St Hilaire 2007 writes arctan(xc/x) = -170.7° [24]. Neither value is reachable by a principal arctangent.

**Direction and duration**
- With τ_X dX/dt ≈ Y and τ_Y dY/dt ≈ -(δ/τ_c)² X [3], the point (X, Y) rotates clockwise. My simulation gives dφ/dt = -0.2618 rad/h (-2π per 24 h).
- Synthesis switches on when φ falls through -1.44 (SynOn), wraps through ±π, and switches off when φ falls through 2.78 (SynOff). That on-arc is 2π - 4.22 = 2.06 rad, about 7.9 h.
- The opposite reading (synthesis on inside the window) would give 16.1 h per day.

**Numerical check**

Method: a scratch script imports the repo unmodified and swaps in `arctan2` for one run.
- Light: 1000 lx from 08:00 to 20:00, 40 lx otherwise.
- Free sleep and the photopic ("2018") photic drive.
- 20 days at 60 s output; the last 3 days are analysed.
- The run fed lux into `melatonin_suppression`, which expects W/m², so it validates timing only, not melatonin amplitude.

| Phase function | Synthesis on | Hours/day | Sleep | DLMO proxy t(ϕ_on)+0.5 h | MELpeak / CBTmin (Eq 16-17) |
|---|---|---|---|---|---|
| `numpy.arctan2(Y, X)` | 20:28 to 04:38 | 8.2 | 23:26 to 08:03 | 20:58 | 03:19 / 05:19 |
| `numpy.atan(Y / X)` (current) | 08:40 to 09:05 and 20:28 to 20:59 | 0.9 | 23:26 to 08:03 | 09:10 and 20:58 | none (no solution) |

- Huang 2026 reports sleep onset 23:40, wake around 08:00 and DLMO 21:40 for the same light schedule [5]. The atan2 run reproduces the sleep timing, but its DLMO proxy is 40 min earlier, which I have not explained.
- Under `atan`, synthesis runs in two short windows per day. That is what `data/original_test.png` shows: A(t) has two narrow spikes per day instead of one plateau.
- `models.test.py::test_circadian_phase` asserts 1.4407 for (X, Y) = (-0.14, -1.07), which encodes the bug. The four-quadrant value is -1.7009.

### Other discrepancies found in the primary sources

Tekieh 2020 here means the accepted manuscript [4]; the version of record was not read.

1. **Melatonin suppression has the wrong functional form.** Tekieh Eq 9 is a Hill function, r = 1 - r_a/(1 + (E/r_b)^(-r_c)). `melatonin_suppression` implements a logistic, 1 - R_A/(1 + exp(-(E - R_B)/R_C)), with the same constants. NIF uses the Hill form [58].

   | E (W/m²) | Repo r | Tekieh r |
   |---|---|---|
   | 0 (darkness) | 0.509 (49% suppression) | 1.0 |
   | 1 | 0.235 | 0.055 |
2. **Wrong time constant τ_A.** Tekieh sets τ_A = 1.5 h for the melanopic model. The repo integrates A with `TAU_ALPHA` = 3.11 h and leaves `TAU_A` = 1.5 h unused. NIF uses 1.5 h "as per Tekieh et al. 2020 (NOT as per Abeysuriya et al. 2018)" [58].
3. **Sigmoid normalisation is unconfirmed.** Tekieh Eq 14, as printed, is a plain logistic described as "normalized to the range between 0 and 1". NIF uses (S(E) - S(0))/(S(1000) - S(0)) after author emails [58], and `models.py` has the equivalent line commented out.
4. **Light gating is unconfirmed for r and Θ_L.** Postnova 2018 Eq 14 gates light (I = I·S) under the photic drive [3]. NIF, like this repo, gates only α and passes ungated light to r and Θ_L [58]. Whether r and Θ_L should see gated light is not stated in what I read. It matters whenever the light series is non-zero during sleep, as in `data/output.csv`.

## Reference outputs for validation

**Published reference trajectories: none found.**
- None of the four papers is in PMC or Europe PMC, and their supplements could not be opened [57].
- Postnova 2018 shows no supplementary-material link on the archived page [3].
- The group's code is "available upon request" [5].
- The Alertness API requires an access request by email. It takes `photopic_lux` only and works at 1 to 15 min resolution [59].
- NIF_Photobiology publishes only aggregated building-simulation outputs (30 min medians and daily summaries), not scenario trajectories [58].
- ModelDB and CellML have only the earlier Phillips-Robinson switch [55].

**Testable published numbers**
- Postnova 2018: entrained sleep 23:30 to 08:00 under 250/40/0 lux [3].
- Knock 2021: about 8.5 h of sleep starting about 23:30 under the same schedule [7].
- Huang 2026: sleep 23:40 to about 08:00 and DLMO 21:40 under 1000/40/0 lux [5].
- Tekieh 2020: KSS, PRC and DRC data sets with NRMSE values [4].

**`data/output.csv`, from the file itself**
- **Shape.** 17,281 rows, 0 to 48 h at a 10 s step. Semicolon-delimited with comma decimals, i.e. a spreadsheet export from a European-locale machine, not output of this repo's Python.
- **No git history.** `data/` is in `.gitignore`.
- **How the repo uses it.** `validation.ipynb` loads it as "Comparison Data". `examples/run.bak.ipynb` hard-codes night and day levels of 27 and 165 with the comment "from the scientist".
- **Structural match: the light levels follow the published default schedule.** The constant levels are 26.9832, 84.32265 and 168.6453, which are exactly 40, 125 and 250 × 0.6745812.
  - The timing fits the Postnova 2018 schedule: 250 from 08:00 to 20:00, 40 otherwise, with ramps through the 125 midpoint at 08:00 and 20:00 [3].
  - On day 1, 09:00 to 16:00 is replaced by a measured profile.
  - The scale factor 0.6745812 is unexplained. It matches no standard mel-DER found by the delegated search.
- **The day-1 segment is the profile in `data/input_irradiance_mel.csv` (2018-06-02, 09:02 to 15:58), scaled.**
  - Its ratio to that file's mEDI column is 1.106 ± 0.011. Equivalently, I_mel ≈ irradiance_mel / 1.2 × 10⁻³.
  - That is consistent with Lucas-style melanopic lux, which is about 1.104 × mEDI (683.002·∫V/∫s_mel = 832.43, computed from CIE tables shipped with luxpy [47]).
  - If so, `validation.ipynb`'s `I_mel * 0.0013262` conversion overstates melanopic irradiance by about 10%.
- **What it does not match.** The 10 s step matches neither NIF (20 s default) nor the Alertness API (1 to 15 min) [58][59].
- **Lead, unverified.** NIF's changelog credits "Dr. Pierson", and this repo's README names Clotilde Pierson as project lead. The file may come from her pipeline. Ask her before using it as ground truth.

## Could not verify

- **Abeysuriya 2018:** full text, including Eq 1-4, Eq 14-15, the synthesis window equation, and the values of τ_A, R_g, U*, ρ_b* and T_U. The repo's `TAU_ALPHA` = 3.11 h is unverified.
  - Lead: Crossref lists an accepted manuscript at https://onlinelibrary.wiley.com/doi/am-pdf/10.1111/jpi.12474 (AM licence from 2019-02-13) [57]. Tekieh 2020's accepted manuscript was readable at the same path type.
  - The publisher's bot check blocked only automated access. The owner can probably open this AM, the two Sage PDFs [1][3] and the Postnova 2018 supplement page in a normal browser.
- **Postnova 2016 and the photic-rate constants:** full text not read [1]. Three group papers print the 2018 form, α0 · I/(I + I1) · √(I/I0) with I0 = 100 lx and I1 = 9500 lx [3][4][5]. Postnova 2013 and St Hilaire 2007 instead put I0 = 9500 inside the power law [6][24]. Without Postnova 2016, I cannot tell whether this is a deliberate recalibration or a label swap that propagated. A library default should flag it.
- **Tekieh 2020:** the version of record, including whether it differs from the accepted manuscript [4].
- **Light gating of r and Θ_L:** whether the Postnova group gates the light that enters suppression r and Θ_L.
- **DLMO offset:** why my atan2 run puts the DLMO proxy 40 min earlier than Huang 2026 [5].
- **Kronauer 1999 papers and the 2000 erratum:** the Process L and pacemaker parameters, and the sign of the x⁷ term as printed in St Hilaire's MathML [21][22][24].
- **Forger 1999 original parameters** [23].
- **Hannay 2019 supplement parameters** [25].
- **Lineage papers:** full text of Phillips & Robinson 2007, Phillips, Chen & Robinson 2010, and Skeldon 2016 [11][12][16].
- **Borbély 1982, Daan 1984 and Åkerstedt 2004:** I relied on the restatements in [19][32].
- **Jewett & Kronauer 1999 alertness equations** [33].
- **Hommes & Giménez 2015 fitted values** [34].
- **SAFTE:** whether it has a light input.
- **CIE S 026:** the toolbox file was not opened. The 1.3262 mW/lm constant is taken from luxpy's source [47], not the standard's text.
- **`data/output.csv`:** its provenance, the 0.6745812 factor, and whether its KSS came from the photopic or the melanopic model.
- **Delegated reading:** the GitHub `main` details of `circadian` (open issues, test contents) and the FIPS, LightLogR, hcrsimpy, HCRSim.jl, Alertness API and ModelDB facts were read by delegated search agents, not by me directly.

## References

1. Postnova S, Lockley SW, Robinson PA. Sleep propensity under forced desynchrony in a model of arousal state dynamics. J Biol Rhythms 2016;31(5):498-508. doi:10.1177/0748730416658806. Not read (publisher bot check).
2. Abeysuriya RG, Lockley SW, Robinson PA, Postnova S. A unified model of melatonin, 6-sulfatoxymelatonin, and sleep dynamics. J Pineal Res 2018;64(4):e12474. doi:10.1111/jpi.12474. Abstract only.
3. Postnova S, Lockley SW, Robinson PA. Prediction of cognitive performance and subjective sleepiness using a model of arousal dynamics. J Biol Rhythms 2018;33(2):203-218. doi:10.1177/0748730418758454. Read via http://web.archive.org/web/20241231060354/https://journals.sagepub.com/doi/10.1177/0748730418758454 (Table 1, Table 3).
4. Tekieh T, Lockley SW, Robinson PA, McCloskey S, Zobaer MS, Postnova S. Modeling melanopsin-mediated effects of light on circadian phase, melatonin suppression, and subjective sleepiness. J Pineal Res 2020;69(3):e12681. doi:10.1111/jpi.12681. Accepted manuscript read via http://web.archive.org/web/20220519191229/https://onlinelibrary.wiley.com/doi/am-pdf/10.1111/jpi.12681.
5. Huang S, Bin YS, Caillaud C, Postnova S. Modeling the effects of flight itinerary on jetlag duration. Sleep 2026;49. doi:10.1093/sleep/zsag063. PMC13267907 (main text Eq 4-8, Supplementary Table S1).
6. Postnova S, Robinson PA, Postnov DD. Adaptation to shift work: physiologically based modeling of the effects of lighting and shifts' start time. PLoS One 2013;8(1):e53379. doi:10.1371/journal.pone.0053379. PMC3537665.
7. Knock SA, Magee M, Stone JE, et al. Prediction of shiftworker alertness, sleep, and circadian phase using a model of arousal dynamics constrained by shift schedules and light exposure. Sleep 2021;44(11):zsab146. doi:10.1093/sleep/zsab146. PMC8598188.
8. Raison S, Sanz-Leon P, Schmidt C, Muto V, Postnova S. A bistable stochastic model quantifies performance degradation during sleep deprivation. Sleep 2025;48. doi:10.1093/sleep/zsaf205. PMC12696388. Code: https://github.com/SR393/PVT-SD-vulnerability
9. Varma P, Postnova S, Knock S, et al. SleepSync: early testing of a personalised sleep-wake management smartphone application for improving sleep and cognitive fitness in defence shift workers. Clocks & Sleep 2024;6(2):267-280. doi:10.3390/clockssleep6020019. PMC11203003.
10. Varma P, Postnova S, Phillips AJK, et al. Pilot feasibility testing of biomathematical model recommendations for personalising sleep timing in shift workers. J Sleep Res 2025;34(4):e14026. doi:10.1111/jsr.14026. PMC12215231.
11. Phillips AJK, Robinson PA. A quantitative model of sleep-wake dynamics based on the physiology of the brainstem ascending arousal system. J Biol Rhythms 2007;22(2):167-179. doi:10.1177/0748730406297512. Abstract only.
12. Phillips AJK, Chen PY, Robinson PA. Probing the mechanisms of chronotype using quantitative modeling. J Biol Rhythms 2010;25(3):217-227. doi:10.1177/0748730410369208. Abstract only.
13. Phillips AJK, et al. Revisiting spontaneous internal desynchrony using a quantitative model of sleep physiology. J Biol Rhythms 2011;26(5):441-453. doi:10.1177/0748730411414163. PMC3557950.
14. Swaminathan K, et al. Are individual differences in sleep and circadian timing amplified by use of artificial light sources? J Biol Rhythms 2017;32(2):165-176. doi:10.1177/0748730417699310. PMC5593073.
15. Phillips AJK, Robinson PA, et al. Mammalian sleep dynamics: how diverse features arise from a common physiological framework. PLoS Comput Biol 2010;6:e1000826. doi:10.1371/journal.pcbi.1000826. PMC2891699.
16. Skeldon AC, Derks G, Dijk DJ. Modelling changes in sleep timing and duration across the lifespan. Sleep Med Rev 2016;28. doi:10.1016/j.smrv.2015.05.011. Abstract only.
17. Skeldon AC, Phillips AJK, Dijk DJ. The effects of self-selected light-dark cycles and social constraints on human sleep and circadian timing: a modeling approach. Sci Rep 2017;7:45158. doi:10.1038/srep45158. PMC5366875 (with supplement).
18. Skeldon AC, et al. Method to determine whether sleep phenotypes are driven by endogenous circadian rhythms or environmental light by combining longitudinal data and personalised mathematical models. PLoS Comput Biol 2023;19:e1011743. doi:10.1371/journal.pcbi.1011743. PMC10817199.
19. Skeldon AC, Dijk DJ. The complexity and commonness of the two-process model of sleep regulation from a mathematical perspective. npj Biol Timing Sleep 2025;2:24. doi:10.1038/s44323-025-00039-z. PMC12176636.
20. Skeldon AC, Dijk DJ, Derks G. Mathematical models for sleep-wake dynamics: comparison of the two-process model and a mutual inhibition neuronal model. PLoS One 2014;9:e103877. doi:10.1371/journal.pone.0103877. PMC4118955.
21. Kronauer RE, Forger DB, Jewett ME. Quantifying human circadian pacemaker response to brief, extended, and repeated light stimuli over the phototopic range. J Biol Rhythms 1999;14(6):500-515. doi:10.1177/074873099129001073. Abstract only.
22. Jewett ME, Forger DB, Kronauer RE. Revised limit cycle oscillator model of human circadian pacemaker. J Biol Rhythms 1999;14(6):493-499. doi:10.1177/074873099129001064. Abstract only.
23. Forger DB, Jewett ME, Kronauer RE. A simpler model of the human circadian pacemaker. J Biol Rhythms 1999;14(6):532-537. doi:10.1177/074873099129000867. Abstract only.
24. St Hilaire MA, et al. Addition of a non-photic component to a light-based mathematical model of the human circadian pacemaker. J Theor Biol 2007;247(4):583-599. doi:10.1016/j.jtbi.2007.04.001. PMC3123888.
25. Hannay KM, Booth V, Forger DB. Macroscopic models for human circadian rhythms. J Biol Rhythms 2019;34(6):658-671. doi:10.1177/0748730419878298. https://par.nsf.gov/servlets/purl/10173129
26. Serkh K, Forger DB. Optimal schedules of light exposure for rapidly correcting circadian misalignment. PLoS Comput Biol 2014;10(4):e1003523. doi:10.1371/journal.pcbi.1003523. PMC3983044.
27. Huang Y, Mayer C, Cheng P, et al. Predicting circadian phase across populations: a comparison of mathematical models and wearable devices. Sleep 2021;44(10):zsab126. doi:10.1093/sleep/zsab126. PMC8503830.
28. Breslow ER, et al. A mathematical model of the circadian phase-shifting effects of exogenous melatonin. J Biol Rhythms 2013;28(1):79-89. doi:10.1177/0748730412468081. Not read; used via [41].
29. Borbély AA. A two process model of sleep regulation. Hum Neurobiol 1982;1(3):195-204. Not read.
30. Daan S, Beersma DG, Borbély AA. Timing of human sleep: recovery process gated by a circadian pacemaker. Am J Physiol 1984;246(2 Pt 2):R161-R183. doi:10.1152/ajpregu.1984.246.2.R161. Not read.
31. Åkerstedt T, Folkard S, Portin C. Predictions from the three-process model of alertness. Aviat Space Environ Med 2004;75(3 Suppl):A75-83. PMID 15018267. Abstract only.
32. Ingre M, et al. Validating and extending the three process model of alertness in airline operations. PLoS One 2014;9:e108679. doi:10.1371/journal.pone.0108679. PMC4203690.
33. Jewett ME, Kronauer RE. Interactive mathematical models of subjective alertness and cognitive throughput in humans. J Biol Rhythms 1999;14(6):588-597. doi:10.1177/074873099129000920. Abstract only.
34. Hommes V, Giménez MC. A revision of existing Karolinska Sleepiness Scale responses to light: a melanopic perspective. Chronobiol Int 2015;32(6):750-756. doi:10.3109/07420528.2015.1043012. Abstract only.
35. Amundadottir ML, et al. A human-centric approach to assess daylight in buildings for non-visual health potential, visual interest and gaze behavior. Build Environ 2017;113:5-21. doi:10.1016/j.buildenv.2016.09.033. Post-print: https://infoscience.epfl.ch/record/221590
36. Phillips AJK, Klerman EB, Butler JP. Modeling the adenosine system as a modulator of cognitive performance and sleep patterns during sleep restriction and recovery. PLoS Comput Biol 2017;13(10):e1005759. doi:10.1371/journal.pcbi.1005759. PMC5675465. Code: https://github.com/ajkphillips/CSR
37. Fulcher BD, Phillips AJK, Robinson PA. Quantitative physiologically based modeling of subjective fatigue during sleep deprivation. J Theor Biol 2010;264(2):407-419. doi:10.1016/j.jtbi.2010.02.028. Abstract only.
38. Stone JE, et al. Who needs bright light and when? Mapping the interactions of lighting environments and individual differences in circadian light sensitivity. J Biol Rhythms 2025;41(2):192-204. doi:10.1177/07487304251391268. PMC12966363.
39. Papatsimpa C, Schlangen LJM, Smolders KCHJ, et al. The interindividual variability of sleep timing and circadian phase in humans is influenced by daytime and evening light conditions. Sci Rep 2021;11:13709. doi:10.1038/s41598-021-92863-z. PMC8249410.
40. Xu M, Papatsimpa C, Schlangen L, et al. Improving adjustment to daylight saving time transitions with light. Sci Rep 2024;14:15001. doi:10.1038/s41598-024-65705-x. PMC11217455.
41. Arcascope `circadian`. https://github.com/Arcascope/circadian (main at db6eb08); PyPI 1.0.3: https://pypi.org/project/circadian/1.0.3/ ; docs: https://arcascope.github.io/circadian/
42. Arcascope hcrsimpy. https://github.com/Arcascope/hcrsimpy
43. Arcascope HCRSim.jl. https://github.com/Arcascope/HCRSim.jl
44. ojwalch/predicting_dlmo. https://github.com/ojwalch/predicting_dlmo
45. FIPS. https://github.com/humanfactors/FIPS
46. LightLogR. https://github.com/tscnlab/LightLogR ; https://cran.r-project.org/package=LightLogR
47. luxpy. https://github.com/ksmet1977/luxpy (`luxpy/toolboxes/photbiochem/alpha_opic_quantities_cie_s026_2018_cie_tn003_2015.py`; `LICENSE.md`, GPLv3)
48. colour-science. https://github.com/colour-science/colour
49. pysilsub. https://github.com/PySilentSubstitution/pysilsub
50. pyplr. https://github.com/PyPlr/cvd_pupillometry
51. CIE S 026 α-opic Toolbox. doi:10.25039/S026.2018.TB
52. luox. https://github.com/luox-app/luox
53. Skeldon HCL model code. https://github.com/anneskeldon/Homeostatic_circadian_light_model-factors_driving_sleep_phenotypes
54. mas1as/Two_Process_Model. https://github.com/mas1as/Two_Process_Model
55. Phillips-Robinson ports: https://github.com/LockyWebb/PR_Sleep_Stan_Models ; ModelDB 247696 (CellML): https://modeldb.science/247696 ; https://github.com/PabRod/sleepR
56. Other small implementations: https://github.com/Mitchell-Griggs/Arousal_Dynamics_Model ; https://github.com/claire-lansonneur/ModelDB_Postnova_et_al_2009 ; https://github.com/brain-modelling-group/infant-sleep-trajectory
57. Availability lookups: Crossref https://api.crossref.org/works/10.1111/jpi.12474 (link and licence fields); Europe PMC REST https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:%2210.1111/jpi.12474%22&format=json&resultType=core (same for the other three DOIs); Unpaywall https://api.unpaywall.org/v2/10.1111/jpi.12474 ; OpenAlex https://api.openalex.org/works?search=unified%20model%20of%20melatonin
58. C38C/NIF_Photobiology. https://github.com/C38C/NIF_Photobiology (branch `master`, `biological_model.py`; `LICENSE`, GPL-3.0)
59. Alertness API. https://www.alertnessapi.com/
