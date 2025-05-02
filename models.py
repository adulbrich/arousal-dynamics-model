"""Arousal dynamics model for sleep-wake regulation"""

import numpy as np
from numpy import sqrt, exp, power, tanh, atan, vectorize
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
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
        constants.V_WE - constants.NU_MV * Q_v - constants.D_M,
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
    D_v = constants.NU_VH * H + constants.NU_VC * C + constants.A_V
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
        alpha = ((constants.ALPHA_0 * IE) / (IE + constants.I_1)) * sqrt(
            IE / constants.I_0
        )
    elif version == "2020":
        alpha = (
            (constants.ALPHA_0 * IE) / (IE + constants.I_1 * constants.F_4100K)
        ) * sqrt(IE / (constants.I_0 * constants.F_4100K))
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

    D_p = alpha * (1 - P) * (1 - constants.EPSILON * X) * (1 - constants.EPSILON * Y)
    return D_p


photic_drive_v = vectorize(photic_drive)


def mean_population_firing_rate(V_i):
    """Mean Population Firing Rate Function
    Inputs:
        V_i: V_v or V_m, mean voltages of the VLPO or MA respectively
    Outputs:
        Q: mean population firing rate
    """

    Q = constants.Q_MAX / (1 + exp((constants.THETA - V_i) / constants.SIGMA_PRIME))
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

    S = 1 / (1 + exp((constants.S_B - E_emel) / constants.S_C))
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

    AM = constants.THETA_0 + (constants.THETA_H + Theta_L) * H + constants.THETA_C * C
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


def circadian_phase(X, Y):
    """Circadian Phase Function
    Inputs:
        X, Y: Circadian Variables
    Outputs:
        phi: circadian phase
    """

    phi = atan(Y / X)
    return phi


circadian_phase_v = vectorize(circadian_phase)


def melatonin_synthesis_regulation(phi):
    """Melatonin Synthesis Regulation Function
    Inputs:
        phi: circadian phase
    Outputs:
        m_phi: melatonin synthesis regulation
    """

    if constants.PHI_ON <= phi <= constants.PHI_OFF:
        m_phi = 0
    else:
        m_phi = 1

    return m_phi


melatonin_synthesis_regulation_v = vectorize(melatonin_synthesis_regulation)

def urinary_excretion_rate(rho_b_vector, t_values, t, delay=constants.T_U):
    """Calculate the urinary excretion rate of aMT6s
    
    Inputs:
        rho_b_vector: vector of aMT6s concentrations in blood
        t_values: corresponding time points for rho_b_vector
        t: current time point to calculate excretion rate for
        delay: time delay in seconds (default from constants.T_U)
        
    Outputs:
        u: urinary excretion rate of aMT6s
    
    The function returns the excretion rate according to:
    u(t) = U_STAR * rho_b(t - T_U) / RHO_B_STAR
    
    If the delayed time point is before the start of the simulation,
    the function will use the earliest available blood concentration.
    """
    # Calculate the delayed time point
    t_delayed = t - delay
    
    # Find the index of the closest time point in t_values 
    # that is not greater than t_delayed
    if t_delayed < t_values[0]:
        # If the delayed time is before simulation start, use the first value
        rho_b_delayed = rho_b_vector[0]
    else:
        # Find the index of the closest time point that's not greater than t_delayed
        idx = np.searchsorted(t_values, t_delayed, side='right') - 1
        idx = max(0, min(idx, len(rho_b_vector) - 1))  # Ensure index is in bounds
        rho_b_delayed = rho_b_vector[idx]
    
    # Calculate urinary excretion rate
    u = constants.U_STAR * rho_b_delayed / constants.RHO_B_STAR
    
    return u

def forced_wake(t, waketime=6, bedtime=22):
    """Forced Wake Function
    Inputs:
        t: time in seconds
        waketime: time to wake up in hours
        bedtime: time to sleep in hours
    Outputs:
        F_w: forced wake, 1 if forced wake, 0 if not
    """
    if (t / 3600 % 24) >= waketime and (t / 3600 % 24) <= bedtime:
        F_w = 1
    else:
        F_w = 0
    return F_w


forced_wake_v = vectorize(forced_wake)


def irradiance(t, input_irradiance, waketime=6, bedtime=22):
    """Irradiance Function
    Inputs:
        t: time in seconds
        input_irradiance: input irradiance data
    Outputs:
        E_emel: Melanopic Irradiance
    """

    t = t % (24 * 3600)

    if t <= waketime * 3600 or t >= bedtime * 3600:
        E_emel = 0
    elif t < 7.5 * 3600:
        E_emel = ((t - waketime * 3600) / 3600 / 1.5 * 50) / constants.CONVERSION_FACTOR
    elif t > 17 * 3600 and t < 19 * 3600:
        E_emel = (50 - ((t - 17 * 3600) / 3600 / 2 * 40)) / constants.CONVERSION_FACTOR
    elif t > 19 * 3600 and t < bedtime * 3600:
        E_emel = 10 / constants.CONVERSION_FACTOR
    else:
        csv_hours = input_irradiance["hours"].values
        csv_irradiance_mel = input_irradiance["irradiance_mel"].values

        interpolator = interp1d(
            csv_hours,
            csv_irradiance_mel,
            kind="cubic",
            bounds_error=False,
            fill_value=(csv_irradiance_mel[0], csv_irradiance_mel[-1]),
        )
        E_emel = interpolator(t / 3600)

    return E_emel


def model(y, t, input_irradiance, waketime, bedtime, minE, maxE, version="2020"):
    V_v, V_m, H, X, Y, P, Theta_L, A, rho_b = y

    IE = irradiance(t, input_irradiance, waketime, bedtime)
    S = state(V_m)
    Sigmoid = (sigmoid(IE) - sigmoid(minE)) / (sigmoid(maxE) - sigmoid(minE))
    alpha = photoreceptor_conversion_rate(IE, S, version)
    Q_m = mean_population_firing_rate(V_m)
    Q_v = mean_population_firing_rate(V_v)
    C = circadian_drive(X, Y)
    D_v = total_sleep_drive(H, C)
    D_n = nonphotic_drive(X, S)
    D_p = photic_drive(X, Y, P, alpha)
    F_w = forced_wake(t, waketime, bedtime)
    W = wake_effort(Q_v, F_w)
    r = melatonin_suppression(IE)
    phi = circadian_phase(X, Y)
    m_phi = melatonin_synthesis_regulation(phi)

    gradient_y = [
        (constants.NU_VM * Q_m - V_v + D_v) / constants.TAU_V,
        (constants.NU_MV * Q_v - V_m + constants.D_M + W) / constants.TAU_M,
        (constants.NU_HM * Q_m - H) / constants.TAU_H,
        (
            Y
            + constants.GAMMA
            * (X / 3.0 + power(X, 3) * 4.0 / 3.0 - power(X, 7) * 256.0 / 105.0)
            + constants.NU_XP * D_p
            + constants.NU_XN * D_n
        )
        / constants.TAU_X,
        (
            D_p * (constants.NU_YY * Y - constants.NU_YX * X)
            - power(
                (constants.DELTA / constants.TAU_C),
                2,
            )
            * X
        )
        / constants.TAU_Y,
        alpha * (1 - P) - (constants.BETA * P),
        (-Theta_L + constants.NU_LA * Sigmoid) / constants.TAU_L,
        (m_phi * r - A) / constants.TAU_ALPHA,
        (constants.U_STAR / constants.R_G) * (A - rho_b / constants.RHO_B_STAR),
    ]
    return gradient_y


# def model_2018(y, t, input_function, forced_wake, minE, maxE):
#     return model(y, t, input_function, forced_wake, minE, maxE, version="2018")
# def model_2020(y, t, input_function, forced_wake, minE, maxE):
#     return model(y, t, input_function, forced_wake, minE, maxE, version="2020")


def model_run(
    days,
    steps,
    input_irradiance,
    waketime=6,
    bedtime=22,
    minE=0,
    maxE=1000,
    debug=False,
):
    """
    Run the model with default parameters.
    """
    # Define initial conditions
    y0 = [-4.55, -0.07, 13.29, -0.14, -1.07, 0.10, -5.00e-06, 0, 25]

    # Define time points
    t = np.linspace(0, days * 24 * 60 * 60, steps)

    version_year = "2020"  # or "2018"

    if debug:
        # Generate irradiance values for all time points
        irr_values = [
            irradiance(time, input_irradiance, waketime, bedtime) for time in t
        ]

        # Convert time from seconds to hours for better readability
        t_hours = t / 3600

        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(t_hours, irr_values)
        plt.title("Irradiance Over Time")
        plt.xlabel("Time (hours)")
        plt.ylabel("Irradiance (melanopic lux)")
        plt.grid(True)

        # Add day markers if simulating multiple days
        if days > 1:
            for day in range(1, days):
                plt.axvline(x=day * 24, color="r", linestyle="--", alpha=0.3)

        # Highlight wake and bedtime
        for day in range(days):
            day_offset = day * 24
            plt.axvline(
                x=day_offset + waketime,
                color="g",
                linestyle=":",
                label="Wake time" if day == 0 else "",
            )
            plt.axvline(
                x=day_offset + bedtime,
                color="b",
                linestyle=":",
                label="Bed time" if day == 0 else "",
            )

        plt.legend()
        plt.show()

    (sol, temp) = odeint(
        model,
        y0,
        t,
        args=(
            input_irradiance,
            waketime,
            bedtime,
            minE,
            maxE,
            version_year,
        ),
        full_output=True,
    )

    return sol, t
