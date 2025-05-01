"""Arousal dynamics model for sleep-wake regulation"""

from numpy import sqrt, exp, power, tanh, vectorize
import constants

def wake_effort(Q_v, forced=0):
    """
    Wake Effort Function
    Inputs:
        Q_v: mean population firing rate of the VLPO
        forced: 1 if forced wake, default 0
    Outputs:
        W: wake effort
    """
    W = forced * max(
        0,
        constants.V_WE
        - constants.NU_MV * Q_v
        - constants.D_M,
    )
    return W


wake_effort_v = vectorize(wake_effort)


def total_sleep_drive(H, C):
    """
    Total Sleep Drive Function
    Inputs:
        H: homeostatic drive
        C: circadian drive, sleep propensity model
    Outputs:
        D_v:  total sleep drive
    """
    D_v = (
        constants.NU_VH * H
        + constants.NU_VC * C
        + constants.A_V
    )
    return D_v


total_sleep_drive_v = vectorize(total_sleep_drive)


def nonphotic_drive(X, S):
    """Nonphotic Drive Function
    Inputs:
        X: Circadian Variables
        S: Wake = 1 or Sleep = 0 state
    Outputs:
        D_n: nonphotic drive to the circadian
    """
    D_n = (S - (2.0 / 3.0)) * (1 - tanh(constants.R * X))
    return D_n

nonphotic_drive_v = vectorize(nonphotic_drive)


def photoreceptor_conversion_rate(IE, S, version="2020"):
    """Photoreceptor Conversion Rate Function
    Inputs:
        IE: Illuminance (lux) I or Melanopic Irradiance E_emel
        S: Wake = 1 or Sleep = 0 state
        version: 2018 uses Illuminance, 2020 uses melanopic irradiance
    Outputs:
        alpha: the photorecpetor conversion rate
    """

    IE = IE * S

    alpha = 0

    if version == "2018":
        alpha = (
            (constants.ALPHA_0 * IE)
            / (IE + constants.I_1)
        ) * sqrt(IE / constants.I_0)
    elif version == "2020":
        alpha = (
            (constants.ALPHA_0 * IE)
            / (IE + constants.I_1 * constants.F_4100K)
        ) * sqrt(
            IE / (constants.I_0 * constants.F_4100K)
        )
    else:
        raise ValueError("Invalid version. Use '2018' or '2020'.")
    return alpha


photoreceptor_conversion_rate_v = vectorize(photoreceptor_conversion_rate)


def photic_drive(X, Y, P, alpha):
    """Photic Drive Function
    Inputs:
        X, Y: Circadian Variables
        P: Photoreceptor Activity
        alpha: photoreceptor conversion rate
    Outputs:
        D_p: photic drive to the circadian
    """

    D_p = (
        alpha
        * (1 - P)
        * (1 - constants.EPSILON * X)
        * (1 - constants.EPSILON * Y)
    )
    return D_p


photic_drive_v = vectorize(photic_drive)


def mean_population_firing_rate(V_i):
    """Mean Population Firing Rate Function
    Inputs:
        V_i: V_v or V_m, mean voltages of the VLPO or MA respectively
    Outputs:
        Q: mean population firing rate
    """

    Q = constants.Q_MAX / (
        1
        + exp(
            (constants.THETA - V_i)
            / constants.SIGMA_PRIME
        )
    )
    return Q

mean_population_firing_rate_v = vectorize(mean_population_firing_rate)

def state(V_m):
    """Wake/Sleep State Function
    Inputs:
        V_m: Mean Voltage of the monoaminergic (MA) wake-active neuronal populations
    Outputs:
        S: sleep state, 1 is awake, 0 is asleep
    """

    if V_m > constants.V_TH:
        S = 1
    else:
        S = 0
    return S

state_v = vectorize(state)

def sigmoid(E_emel):
    """Sigmoid Function
    Inputs:
        E_emel: Melanopic Irradiance
    Outputs:
        S: sigmoid in range [0 1]
    """

    S = 1 / (
        1
        + exp(
            (constants.S_B - E_emel)
            / constants.S_C
        )
    )
    return S

sigmoid_v = vectorize(sigmoid)

def alertness_measure(C, H, Theta_L=0):
    """Alertness Measure Function
    Inputs:
        C: circadian drive, sleep propensity model
        H: homeostatic drive
        Theta_L: light-dependent modulation of the homeostatic weight
    Outputs:
        AM: alertness measure on the KSS
    """

    AM = (
        constants.THETA_0
        + (constants.THETA_H + Theta_L) * H
        + constants.THETA_C * C
    )
    return AM


alertness_measure_v = vectorize(alertness_measure)


def circadian_drive(X, Y):
    """Circadian Drive Function
    Inputs:
        X, Y: Circadian Variables
    Outputs:
        C: circadian drive
    """

    C = 0.1 * ((1.0 + X) / 2.0) + power(
        ((3.1 * X - 2.5 * Y + 4.2) / (3.7 * (X + 2))), 2
    )
    return C

circadian_drive_v = vectorize(circadian_drive)

def melatonin_suppression(E_emel):
    """Melatonin Suppression Function
    Inputs:
        E_emel: Melanopic Irradiance
    Outputs:
        r: melatonin suppression
    """

    r = 1 - (
        constants.R_A
        / (
            1
            + power(
                E_emel / constants.R_B,
                -constants.R_C,
            )
        )
    )
    return r


melatonin_suppression_v = vectorize(melatonin_suppression)


def model(y, t, input_function, forced_wake, minE, maxE, version="2020"):
    V_v, V_m, H, X, Y, P, Theta_L = y

    IE = input_function(t)
    S = state(V_m)
    Sigmoid = (sigmoid(IE) - sigmoid(minE)) / (sigmoid(maxE) - sigmoid(minE))
    alpha = photoreceptor_conversion_rate(IE, S, version)
    Q_m = mean_population_firing_rate(V_m)
    Q_v = mean_population_firing_rate(V_v)
    C = circadian_drive(X, Y)
    D_v = total_sleep_drive(H, C)
    D_n = nonphotic_drive(X, S)
    D_p = photic_drive(X, Y, P, alpha)
    F_w = forced_wake(t)
    W = wake_effort(Q_v, F_w)

    gradient_y = [
        (constants.NU_VM * Q_m - V_v + D_v)
        / constants.TAU_V,
        (
            constants.NU_MV * Q_v
            - V_m
            + constants.D_M
            + W
        )
        / constants.TAU_M,
        (constants.NU_HM * Q_m - H)
        / constants.TAU_H,
        (
            Y
            + constants.GAMMA
            * (X / 3.0 + power(X, 3) * 4.0 / 3.0 - power(X, 7) * 256.0 / 105.0)
            + constants.NU_XP * D_p
            + constants.NU_XN * D_n
        )
        / constants.TAU_X,
        (
            D_p
            * (
                constants.NU_YY * Y
                - constants.NU_YX * X
            )
            - power(
                (
                    constants.DELTA
                    / constants.TAU_C
                ),
                2,
            )
            * X
        )
        / constants.TAU_Y,
        alpha * (1 - P) - (constants.BETA * P),
        (-Theta_L + constants.NU_LA * Sigmoid)
        / constants.TAU_L,
    ]
    return gradient_y


def model_2018(y, t, input_function, forced_wake, minE, maxE):
    return model(y, t, input_function, forced_wake, minE, maxE, version="2018")


def model_2020(y, t, input_function, forced_wake, minE, maxE):
    return model(y, t, input_function, forced_wake, minE, maxE, version="2020")
