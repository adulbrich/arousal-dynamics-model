"""Constants for arousal dynamics model.

Contains all constants used in the model organized by category.
References: Postnova et al. 2018, Tekieh et al. 2020
"""

from numpy import pi

CONVERSION_FACTOR = 754.0340838

# Time constants for model (in seconds)
TAU_V = 50.0  # s
TAU_M = TAU_V
TAU_H = 59.0 * 3600.0  # s
TAU_X = (24.0 * 3600.0) / (2.0 * pi)  # s
TAU_Y = TAU_X
TAU_C = 24.2 * 3600.0  # s
TAU_A = 1.5 * 3600.0  # s  # 1.5 hours - Tekieh et al. 2020
TAU_L = 24.0 * 60.0  # s  # 24 min - Tekieh et al. 2020
TAU_ALPHA = 3.11 * 3600 # s

# Coupling strengths constants
NU_VM = -2.1  # mV
NU_MV = -1.8  # mV
NU_HM = 4.57  # s
NU_XP = 37.0 * 60.0  # s
NU_XN = 0.032
NU_YY = (1.0 / 3.0) * 37.0 * 60.0  # calculated from NU_XP
NU_YX = 0.55 * 37.0 * 60.0  # calculated from NU_XP
NU_VH = 1.0
NU_VC = -0.5  # mV
NU_LA = -0.11  # Tekieh et al. 2020

# Circadian rhythm constants
GAMMA = 0.13
DELTA = 24.0 * 3600.0 / 0.99729  # s
BETA = 0.007 / 60.0  # s^-1

# External neuronal drives constants
D_M = 1.3  # mV
A_V = -10.3  # mV

# Wake effort constants
V_WE = -0.07  # mV
V_TH = -2.0  # mV

# Firing rate constants
Q_MAX = 100.0  # s^-1
THETA = 10.0  # mV
SIGMA_PRIME = 3.0  # mV

# Photic drive constants
EPSILON = 0.4
I_0 = 100  # lx
I_1 = 9500  # lx
ALPHA_0 = 0.1 / 60.0  # s^-1
F_4100K = 8.19e-4  # Tekieh et al. 2020

# Non-photic drive constants
R = 10.0

# Sigmoid function parameters
S_B = 0.05  # W/m^2
S_C = 1 / 223.5  # m^2/W

# Melatonin suppression parameters
R_A = 1
R_B = 0.031  # W/m^2
R_C = 0.82

# Melatonin synthesis
PHI_ON = -1.44 # rad, SynOn
PHI_OFF = 2.78 # rad, SynOff
U_STAR = 0.47 # pmol/L/s, urine peak
A_0 = U_STAR  # pmol/L/s, melatonin peak synthesis rate
R_G = 0.9
RHO_B_STAR = 325 # pmol/L, plasma peak
T_U = 0.96 * 3600 # s, urine time lag

# KSS parameters
THETA_0 = -24.34
THETA_H = 2.28
THETA_C = -1.74
