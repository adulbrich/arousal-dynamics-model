"""The light-driven arousal dynamics model.

Sleep/wake switch, homeostatic and circadian drives after Postnova et al. 2016
and 2018 ([P16], [P18]); melatonin and urinary aMT6s after Abeysuriya et al.
2018 ([A18]); melanopic light input and the light-dependent alertness term after
Tekieh et al. 2020 ([T20]). Citations are in the README.

Every equation is a function of state values and an `ArousalParameters`, works
on scalars and arrays alike, and names its source. Known discrepancies with the
papers are GitHub issues, indexed in issue #7 and cited below by number. The
module began as an exact copy of the original code; an equation that departs
from it says so and cites its issue, and each fix lands on its own so its
effect on the golden outputs is visible.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy import exp, power, sqrt, tanh

from arousal_dynamics.inputs import Inputs
from arousal_dynamics.units import SECONDS_PER_DAY, FloatArray

type Value = Any
"""A float or a NumPy array: every equation broadcasts."""


def _p(default: float, unit: str, source: str) -> Any:
    return field(default=default, metadata={"unit": unit, "source": source})


@dataclass(frozen=True, kw_only=True)
class ArousalParameters:
    """Every constant of the model, with the papers' values as defaults.

    Two defaults are fitted rather than printed: S_c and theta_H, fitted to the
    KSS the original authors shared (`tests/test_validation.py`).
    `ArousalParameters.papers()` holds the printed values for both.

    Override one with `dataclasses.replace(params, s_b=0.3)`. Derived defaults
    (`tau_m`, `tau_x`, `tau_y`, `nu_yy`, `nu_yx`) are fixed numbers, not
    formulas: overriding `nu_xp` does not move `nu_yy`.

    `field.metadata` carries each parameter's unit and source:
    `dataclasses.fields(ArousalParameters)[0].metadata["unit"]`.
    """

    # Time constants
    tau_v: float = _p(50.0, "s", "P18 Table 1")
    tau_m: float = _p(50.0, "s", "P18 Table 1, equal to tau_v")
    tau_h: float = _p(59.0 * 3600.0, "s", "P18 Table 1")
    tau_x: float = _p((24.0 * 3600.0) / (2.0 * np.pi), "s", "P18 Table 1")
    tau_y: float = _p((24.0 * 3600.0) / (2.0 * np.pi), "s", "P18 Table 1, equal to tau_x")
    tau_c: float = _p(24.2 * 3600.0, "s", "P18 Table 1")
    tau_l: float = _p(24.0 * 60.0, "s", "T20 p. 7")
    tau_alpha: float = _p(
        3.11 * 3600,
        "s",
        "A18 Table 1. T20 uses 1.5 h with its Hill-form suppression (issue #12)",
    )

    # Coupling strengths
    nu_vm: float = _p(-2.1, "mV s", "P18 Table 1")
    nu_mv: float = _p(-1.8, "mV s", "P18 Table 1")
    nu_hm: float = _p(4.57, "nM s", "P18 Table 1")
    nu_xp: float = _p(37.0 * 60.0, "s", "P18 Table 1")
    nu_xn: float = _p(0.032, "1", "P18 Table 1")
    nu_yy: float = _p((1.0 / 3.0) * 37.0 * 60.0, "s", "P18 Table 1, nu_xp / 3")
    nu_yx: float = _p(0.55 * 37.0 * 60.0, "s", "P18 Table 1, 0.55 nu_xp")
    nu_vh: float = _p(1.0, "mV/nM", "P18 Table 1")
    nu_vc: float = _p(-0.5, "mV", "P18 Table 1")
    nu_la: float = _p(-0.11, "1", "T20 Eq 13")

    # Circadian oscillator
    gamma: float = _p(0.13, "1", "P18 Table 1")
    delta: float = _p(24.0 * 3600.0 / 0.99729, "s", "P18 Table 1")
    beta: float = _p(0.007 / 60.0, "1/s", "P18 Table 1")

    # External drives
    d_m: float = _p(1.3, "mV", "P18 Table 1")
    a_v: float = _p(-10.3, "mV", "P18 Table 1")

    # Wake effort
    v_we: float = _p(-0.07, "mV", "P18 Table 1. P16 states +0.07 (issue #13)")
    v_th: float = _p(-2.0, "mV", "P18 Table 1")

    # Firing rate
    q_max: float = _p(100.0, "1/s", "P18 Table 1")
    theta: float = _p(10.0, "mV", "P18 Table 1")
    sigma_prime: float = _p(3.0, "mV", "P18 Table 1")

    # Photic drive
    epsilon: float = _p(0.4, "1", "P18 Table 1")
    i_0: float = _p(100, "lx", "P18 Table 1; the saturation constant of SH07 Eq 9 (issue #6)")
    i_1: float = _p(9500, "lx", "P18 Table 1; the power-law constant of SH07 Eq 1 (issue #6)")
    alpha_0: float = _p(0.1 / 60.0, "1/s", "P18 Table 1")
    f_4100k: float = _p(8.19e-4, "W/m^2/lx", "T20 Eq 7")
    photic_version: Literal["2018", "2020"] = field(
        default="2020",
        metadata={
            "unit": "",
            "source": "2018: photopic lux, P18 Eq 13; 2020: melanopic W/m^2, T20 Eq 7",
        },
    )

    # Non-photic drive
    r: float = _p(10.0, "1", "P18 Table 1")

    # Alertness sigmoid
    s_b: float = _p(0.05, "W/m^2", "T20 p. 7")
    # Fitted, not a paper value. T20 p. 7 prints S_c = 223.5 m^2/W, which the
    # authors gave in 2021 as 1/223.5 W/m^2 (issue #10). That width reproduces
    # neither T20's own dose-response (Fig 6) nor the KSS the authors shared
    # (data/output.csv); 0.0306 W/m^2 is the fit to the shared output, jointly
    # with theta_H, with S_b, nu_LA and tau_L at T20's values
    # (tests/test_validation.py). Fig 6 implies a width of the same order
    # (issue #11).
    s_c: float = _p(
        0.0306,
        "W/m^2",
        "Fitted to the authors' shared output; T20 p. 7 prints 223.5 m^2/W (issues #10, #11)",
    )

    # Melatonin suppression
    r_a: float = _p(1, "1", "T20 Eq 9")
    r_b: float = _p(0.031, "W/m^2", "T20 Eq 9")
    r_c: float = _p(0.82, "1", "T20 Eq 9")

    # Melatonin synthesis and excretion
    phi_on: float = _p(-1.44, "rad", "A18 Table 1, SynOn")
    phi_off: float = _p(2.78, "rad", "A18 Table 1, SynOff")
    u_star: float = _p(0.47, "pmol/L/s", "A18 Table 1, urine peak")
    r_g: float = _p(0.9, "1", "A18 Table 1")
    rho_b_star: float = _p(325, "pmol/L", "A18 Table 1, plasma peak")
    t_u: float = _p(0.96 * 3600, "s", "A18 Table 1, urine time lag")

    # KSS
    theta_0: float = _p(-24.34, "1", "P18 Table 3")
    # Fitted, not the printed value. P18 Table 3 prints 2.28, and 2.284 rounds
    # to it. Fitted jointly with S_c, it removes a constant 0.05 point offset
    # against the KSS the authors shared (tests/test_validation.py).
    theta_h: float = _p(
        2.284, "1/nM", "Fitted to the authors' shared output; P18 Table 3 prints 2.28"
    )
    theta_c: float = _p(-1.74, "1", "P18 Table 3")

    @classmethod
    def papers(cls) -> "ArousalParameters":
        """Every constant as the papers print it, S_c and theta_H included.

        The defaults fit those two to the KSS the original authors shared. Here
        S_c is T20's 223.5 m^2/W, which the authors gave in 2021 as 1/223.5 W/m^2
        (issue #10), and theta_H is P18 Table 3's 2.28. The equations are the
        same for both sets, including the fixes for issues #6, #11 and #14.
        """
        return cls(s_c=1.0 / 223.5, theta_h=2.28)


# Equations. Each keeps the original expression and operation order unless its
# docstring cites a fix, so the golden tests pin results bitwise.


def wake_effort(Q_v: Value, forced: Value, p: ArousalParameters) -> Value:
    """Wake effort W while wake is forced. P18 Table 1 Eq 8."""
    return forced * np.maximum(0, p.v_we - p.nu_mv * Q_v - p.d_m)


def total_sleep_drive(H: Value, C: Value, p: ArousalParameters) -> Value:
    """Total sleep drive D_v from the homeostatic and circadian drives. P18 Table 1 Eq 9."""
    return p.nu_vh * H + p.nu_vc * C + p.a_v


def nonphotic_drive(X: Value, S: Value, p: ArousalParameters) -> Value:
    """Non-photic drive D_n to the circadian oscillator. P18 Table 1 Eq 11."""
    return (S - (2.0 / 3.0)) * (1 - tanh(p.r * X))


def photoreceptor_conversion_rate(IE: Value, S: Value, p: ArousalParameters) -> Value:
    """Photoreceptor activation rate alpha, gated by wake.

    alpha = alpha_0 (I / I_1)^(1/2) I / (I + I_0), St Hilaire et al. 2007
    ([SH07], J Theor Biol 247:583) Eq 9: the power law of their Eq 1, with
    I_1 = 9500 lux, times a factor that is linear below I_0 = 100 lux and
    saturates above it. P18 Table 1 Eq 13 and T20 Eq 6 print I_0 and I_1 the
    other way round (issue #6). P16 (p. 500) builds the light response on
    SH07, and only SH07's form gives the entrained sleep from about 23:30 to
    08:00 that P16 (p. 500), P18 (p. 204) and T20 (p. 7) report for their
    250/40/0 lux protocol; the printed form sleeps from 22:30 to 07:00.

    `p.photic_version` picks the unit: "2018" takes photopic illuminance in
    lux, "2020" melanopic irradiance in W/m^2 with I_0 and I_1 scaled by
    F_4100K (T20 Eq 7).
    """
    IE = IE * S
    if p.photic_version == "2018":
        return p.alpha_0 * sqrt(IE / p.i_1) * IE / (IE + p.i_0)
    if p.photic_version == "2020":
        i_0, i_1 = p.i_0 * p.f_4100k, p.i_1 * p.f_4100k
        return p.alpha_0 * sqrt(IE / i_1) * IE / (IE + i_0)
    msg = f"photic_version must be '2018' or '2020', got {p.photic_version!r}"
    raise ValueError(msg)


def photic_drive(X: Value, Y: Value, P: Value, alpha: Value, p: ArousalParameters) -> Value:
    """Photic drive D_p to the circadian oscillator. P18 Table 1 Eq 12."""
    return alpha * (1 - P) * (1 - p.epsilon * X) * (1 - p.epsilon * Y)


def firing_rate(V: Value, p: ArousalParameters) -> Value:
    """Mean population firing rate Q of the VLPO or MA group. P18 Table 1 Eq 7."""
    return p.q_max / (1 + exp((p.theta - V) / p.sigma_prime))


def sleep_state(V_m: Value, p: ArousalParameters) -> Value:
    """Wake (1) or sleep (0) from the MA mean voltage. P18 Table 1 Eq 15."""
    return np.where(V_m > p.v_th, 1, 0)


def alertness_sigmoid(E_mel: Value, p: ArousalParameters) -> Value:
    """Light-dependent sigmoid driving Theta_L, normalized to [0, 1]. T20 Eq 14.

    T20 p. 7 calls S "a sigmoid function normalized to the range between 0 and
    1": the logistic 1 / (1 + exp((S_b - E) / S_c)), less its value in
    darkness and rescaled, so that S is 0 at E = 0 and tends to 1 in bright
    light (issue #11). The original code normalized it this way until 4ca72b2.
    """
    dark = 1 / (1 + exp(p.s_b / p.s_c))
    return (1 / (1 + exp((p.s_b - E_mel) / p.s_c)) - dark) / (1 - dark)


def kss(C: Value, H: Value, Theta_L: Value, p: ArousalParameters) -> Value:
    """Karolinska Sleepiness Scale from the drives and the light term. T20 Eq 12."""
    return p.theta_0 + (p.theta_h + Theta_L) * H + p.theta_c * C


def circadian_drive(X: Value, Y: Value) -> Value:
    """Circadian drive C to sleep propensity. P18 Table 1 Eq 10."""
    return 0.1 * ((1.0 + X) / 2.0) + power(((3.1 * X - 2.5 * Y + 4.2) / (3.7 * (X + 2))), 2)


def melatonin_suppression(E_mel: Value, p: ArousalParameters) -> Value:
    """Acute suppression factor r of melatonin synthesis by light.

    The logistic form of A18 Eq 4 with the Hill-form parameters of T20 Eq 9.
    That pairing gives r = 0.509 in darkness, where both papers have r = 1
    (issue #9).
    """
    return 1 - p.r_a / (1 + exp(-(E_mel - p.r_b) / p.r_c))


def circadian_phase(X: Value, Y: Value) -> Value:
    """Circadian phase phi from the oscillator state. P18 Table 1 Eq 17.

    Two-quadrant `arctan(Y / X)`, as the papers print it, so phi stays in
    (-pi/2, pi/2). The synthesis window [phi_on, phi_off] needs the full circle
    from a four-quadrant arctangent (issue #8).
    """
    return np.arctan(Y / X)


def melatonin_synthesis(phi: Value, p: ArousalParameters) -> Value:
    """Synthesis switch m: 0 inside [phi_on, phi_off], 1 outside. A18 Eq 2."""
    return np.where((p.phi_on <= phi) & (phi <= p.phi_off), 0, 1)


def urinary_excretion(rho_b: FloatArray, t: FloatArray, p: ArousalParameters) -> FloatArray:
    """Urinary aMT6s excretion rate u(t) = u* rho_b(t - T_U) / rho_b*. A18 Eq 15.

    `rho_b` is read at the last sample at or before t - T_U, and at the first
    sample when t - T_U precedes the trajectory.
    """
    rho_b = np.asarray(rho_b, dtype=float)
    t = np.asarray(t, dtype=float)
    idx = np.clip(np.searchsorted(t, t - p.t_u, side="right") - 1, 0, rho_b.size - 1)
    delayed = np.where(t - p.t_u < t[0], rho_b[0], rho_b[idx])
    return p.u_star * delayed / p.rho_b_star


STATE_NAMES = ("V_v", "V_m", "H", "X", "Y", "P", "Theta_L", "A", "rho_b")


class ArousalModel:
    """The nine-state arousal dynamics model, for `arousal_dynamics.simulate`.

    States, in order: V_v and V_m (VLPO and MA mean voltages, mV), H
    (homeostatic pressure, nM), X and Y (circadian oscillator), P (photoreceptor
    activity), Theta_L (light-dependent alertness term), A (melatonin synthesis
    drive) and rho_b (plasma melatonin, pmol/L). Time is in seconds.
    """

    name = "arousal"
    state_names = STATE_NAMES
    time_unit = "s"

    def default_parameters(self) -> ArousalParameters:
        """Return the defaults: the papers' values, S_c and theta_H fitted."""
        return ArousalParameters()

    def initial_state(self, params: ArousalParameters) -> FloatArray:  # noqa: ARG002
        """Return the starting state of the original `model_run`, found by experiment."""
        return np.array([-4.55, -0.07, 13.29, -0.14, -1.07, 0.10, 0, 0, 25], dtype=float)

    def derivatives(
        self, t: float, y: FloatArray, params: ArousalParameters, inputs: Inputs
    ) -> list[Value]:
        """Right-hand side dy/dt of the model at time `t` in seconds."""
        p = params
        V_v, V_m, H, X, Y, P, Theta_L, A, rho_b = y

        IE = inputs.light(t)
        S = sleep_state(V_m, p)
        # Closed eyes: the alerting term sees light only while awake, as the
        # photic drive does (P18 Table 1 Eq 14, I = I S). The authors' shared
        # output starts the alerting response at wake, not at lights-on (issue #14).
        sigmoid = alertness_sigmoid(IE * S, p)
        alpha = photoreceptor_conversion_rate(IE, S, p)
        Q_m = firing_rate(V_m, p)
        Q_v = firing_rate(V_v, p)
        C = circadian_drive(X, Y)
        D_v = total_sleep_drive(H, C, p)
        D_n = nonphotic_drive(X, S, p)
        D_p = photic_drive(X, Y, P, alpha, p)
        F_w = 0 if inputs.schedule is None else inputs.schedule.forced_wake(t)
        W = wake_effort(Q_v, F_w, p)
        r = melatonin_suppression(IE, p)
        phi = circadian_phase(X, Y)
        m_phi = melatonin_synthesis(phi, p)

        return [
            (p.nu_vm * Q_m - V_v + D_v) / p.tau_v,  # P18 Table 1 Eq 1
            (p.nu_mv * Q_v - V_m + p.d_m + W) / p.tau_m,  # P18 Table 1 Eq 2
            (p.nu_hm * Q_m - H) / p.tau_h,  # P18 Table 1 Eq 3
            (  # P18 Table 1 Eq 4
                Y
                + p.gamma * (X / 3.0 + power(X, 3) * 4.0 / 3.0 - power(X, 7) * 256.0 / 105.0)
                + p.nu_xp * D_p
                + p.nu_xn * D_n
            )
            / p.tau_x,
            (D_p * (p.nu_yy * Y - p.nu_yx * X) - power((p.delta / p.tau_c), 2) * X)
            / p.tau_y,  # P18 Table 1 Eq 5
            alpha * (1 - P) - (p.beta * P),  # P18 Table 1 Eq 6
            (-Theta_L + p.nu_la * sigmoid) / p.tau_l,  # T20 Eq 13
            (m_phi * r - A) / p.tau_alpha,  # A18 Eq 1
            (p.u_star / p.r_g) * (A - rho_b / p.rho_b_star),  # A18 Eq 14
        ]

    def outputs(
        self, t: FloatArray, states: FloatArray, params: ArousalParameters, inputs: Inputs
    ) -> dict[str, FloatArray]:
        """Derive series over a whole trajectory.

        light (W/m^2), S (1 awake), C (circadian drive), KSS, phi (rad) and
        aMT6s (urinary excretion rate).
        """
        p = params
        V_m, H, X, Y, Theta_L, rho_b = (states[:, i] for i in (1, 2, 3, 4, 6, 8))
        C = circadian_drive(X, Y)
        return {
            "light": inputs.light.values(t),
            "S": sleep_state(V_m, p).astype(float),
            "C": C,
            "KSS": kss(C, H, Theta_L, p),
            "phi": circadian_phase(X, Y),
            "aMT6s": urinary_excretion(rho_b, t, p),
        }


def time_grid(days: float, steps: int) -> FloatArray:
    """Return `steps` even times from 0 to `days`, in seconds, as `model_run` did."""
    return np.linspace(0, days * SECONDS_PER_DAY, steps)


__all__ = [
    "STATE_NAMES",
    "ArousalModel",
    "ArousalParameters",
    "alertness_sigmoid",
    "circadian_drive",
    "circadian_phase",
    "firing_rate",
    "kss",
    "melatonin_suppression",
    "melatonin_synthesis",
    "nonphotic_drive",
    "photic_drive",
    "photoreceptor_conversion_rate",
    "sleep_state",
    "time_grid",
    "total_sleep_drive",
    "urinary_excretion",
    "wake_effort",
]
