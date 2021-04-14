
from numpy import sqrt, exp, pi, power, tanh, vectorize

# time constants for model: Postnova et al. 2018 - Table 1
tau_v = 50.0 #s
tau_m = tau_v
tau_H = 59.0*3600.0 #s
tau_X = (24.0*3600.0) / (2.0*pi) #s
tau_Y = tau_X
tau_C = 24.2*3600.0 #s
tau_A = 1.5*3600.0 #s # 1.5 hours # Tekieh et al. 2020 - Section 2.3.2, after Equation 9
tau_L = 24.0*60.0 #s # 24 min # Tekieh et al. 2020 - Section 3.3

# coupling strengths constants: Postnova et al. 2018 - Table 1
nu_vm = -2.1 #mV
nu_mv = -1.8 #mV
nu_Hm = 4.57 #s
nu_Xp = 37.0*60.0 #s
nu_Xn = 0.032
nu_YY = (1.0/3.0)*nu_Xp
nu_YX = 0.55*nu_Xp
nu_vH = 1.0
nu_vC = -0.5 #mV
nu_LA = -0.11 # Tekieh et al. 2020 - Section 3.3
# nu_LA = -0.4 # testing: good results with -0.4

# circadian constants: Postnova et al. 2018 - Table 1
gamma = 0.13
delta = 24.0*3600.0/0.99729 #s
beta = 0.007/60.0 #sˆ-1

#mV # external neuronal drives constants: Postnova et al. 2018 - Table 1
D_m = 1.3 

def wake_effort(Q_v, forced = 0):
    # Wake Effort Function
    # Inputs:
    #   Q_v:  mean population firing rate of the VLPO
    #   forced: 1 if forced wake, default 0
    # Outpus:
    #   W: wake effort

    V_WE = -0.07 #mv # wake effort constants: Postnova et al. 2018 - Table 1

    W = forced * max(0, V_WE-nu_mv*Q_v-D_m) # Postnova et al. 2018 - Table 1, Equation 8
    return W

wake_effort_v = vectorize(wake_effort)

def total_sleep_drive(H,C):
    # Total Sleep Drive Function
    # Inputs:
    #   H: homeostatic drive
    #   C: circadian drive, sleep propensity model
    # Outputs:
    #   D_v:  total sleep drive

    A_v = -10.3 #mV # external neuronal drives constants: Postnova et al. 2018 - Table 1

    D_v = nu_vH*H + nu_vC*C + A_v # Postnova et al. 2018 - Table 1, Equation 9
    return D_v

total_sleep_drive_v = vectorize(total_sleep_drive)

def nonphotic_drive(X, S):
    # Nonphotic Drive to the Circadian Function
    # Inputs:
    #   X: Circadian Variables
    #   S: Wake = 1 or Sleep = 0 state 
    # Outputs:
    #   D_n: nonphotic drive to the circadian

    r = 10.0 # nonphotic drive constant: Postnova et al. 2018 - Table 1

    D_n = (S-(2.0/3.0))*(1-tanh(r*X)) # Postnova et al. 2018 - Table 1, Equation 11
    return D_n

nonphotic_drive_v = vectorize(nonphotic_drive)

def photoreceptor_conversion_rate(IE, S, version = '2020'): 
    # Photoreceptor Conversion Rate Function
    # Inputs:
    #   I or E_emel: Illuminance (lux) or Melanopic Irradiance # Is it melanopic illuminance I_mel instead of just I?
    #   S: Wake = 1 or Sleep = 0 state 
    #   version: 2018 uses Illuminance, 2020 uses melanopic irradiance
    # Output:
    #   alpha: the photorecpetor conversion rate

    IE = IE*S # Postnova et al. 2018 - Table 1, Equation 14
    
    # photic drive constants: Postnova et al. 2018 - Table 1
    I_0 = 100 #lx
    I_1 = 9500 #lx
    alpha_0 = 0.1/60.0 #sˆ-1

    if (version == '2018'):
        alpha = ((alpha_0*IE)/(IE+I_1))*sqrt(IE/I_0) # Postnova et al. 2018 - Table 1, Equation 13
    
    if (version == '2020'):
        F_4100K = 8.19e-4 # Tekieh et al. 2020 - Equation 5
        alpha = ((alpha_0*IE)/(IE+I_1*F_4100K))*sqrt(IE/(I_0*F_4100K)) # Tekieh et al. 2020 - Equation 7

    return alpha

photoreceptor_conversion_rate_v = vectorize(photoreceptor_conversion_rate)

def photic_drive(X, Y, P, alpha):
    # Photic Drive to the Circadian function
    # Inputs:
    #   X, Y: Circadian Variables
    #   P: Photoreceptor Activity
    #   alpha: photoreceptor conversion rate
    # Outputs:
    #   D_p: photic drive to the circadian

    epsilon = 0.4 # photic drive constants: Postnova et al. 2018 - Table 1

    D_p = alpha*(1-P)*(1-epsilon*X)*(1-epsilon*Y) # Postnova et al. 2018 - Table 1, Equation 12
    return D_p

photic_drive_v = vectorize(photic_drive)

def mean_population_firing_rate(V_i):
    # Mean Population Firing Rate Function
    # Inputs:
    #   V_v or V_m: Mean voltages of the VLPO and MA respectively
    # Output:
    #   Q: mean population firing rate

    # firing rate constants: Postnova et al. 2018 - Table 1
    Q_max = 100.0 #sˆ-1
    theta = 10.0 #mV
    sigma_prime = 3.0 #mV

    Q = Q_max / (1 + exp((theta-V_i)/sigma_prime)) # Postnova et al. 2018 - Table 1, Equation 7
    return Q

mean_population_firing_rate_v = vectorize(mean_population_firing_rate)

def state(V_m): 
    # Wake/Sleep State Function
    # Postnova et al. 2018 - Table 1, Equation 15
    # Input:
    #   V_m: Mean Voltage of the monoaminergic (MA) wake-active neuronal populations
    # Output:
    #   S: sleep state, 1 is awake, 0 is asleep 

    V_th = -2.0 #mV # wake effort constants: Postnova et al. 2018 - Table 1

    if (V_m > V_th):
        S = 1
    else:
        S = 0
    return S

state_v = vectorize(state)

def sigmoid(E_emel, version = '2020'):
    # sigmoid function
    # Inputs:
    #   E_emel: Melanopic Irradiance
    # Outputs:
    #   S: sigmoid in range [0 1], is it always so? yes

    # parameters defining the melanopic irradiance value at half-maximal alerting effect and the steepness of the curve
    # Tekieh et al. 2020 - Section 2.3.3
    S_b = 0.05 # W/mˆ2
    S_c = 223.5 # mˆ2/W

    if (version == '2020'):
        # sigmoig was defined for illuminance so we need to convert to illuminance for irradiance input in 2020 model
        E_emel = E_emel/0.0013262 

    S = 1/(1 + exp((S_b-E_emel)/S_c) ) # Tekieh et al. 2020 - Equation 14
    return S

sigmoid_v = vectorize(sigmoid)

def alertness_measure(C, H, Theta_L = 0):
    # Alertness Measure Function
    # Inputs:
    #   H: homeostatic drive
    #   C: circadian drive, sleep propensity model
    #   Tetha_L: light-dependent modulation of the homeostatic weight
    # Outputs:
    #   AM: alertness measure on the KSS

    # KSS: Karolinska Sleepiness Scale
    # Ranges from 1 = "Extremely alert" to 9 = "Extremely sleepy, fighting sleep."
    # KSS default parameters: Postnova et al. 2018 - Table 3
    Theta_0 = -24.34
    Theta_H =   2.28
    Theta_C =  -1.74

    AM = Theta_0 + (Theta_H + Theta_L)*H + Theta_C*C # Postnova et al. 2018 - Equation 23, Tekieh et al. 2020 - Equation 12
    return AM

alertness_measure_v = vectorize(alertness_measure)

def circadian_drive(X,Y):
    # Circadian Drive Function, sleep propensity model
    # Inputs:
    #   X, Y: Circadian Variables
    # Outputs:
    #   C: circadian drive

    C = 0.1*((1.0+X)/2.0)+power(((3.1*X - 2.5*Y + 4.2)/(3.7*(X+2))),2) # Postnova et al. 2016 - Equations 1, 2, and 3
    return C

circadian_drive_v = vectorize(circadian_drive)

def melatonin_suppression(E_emel):
    # Melatonin Suppression Function
    # Inputs:
    #   E_emel: Melanopic Irradiance
    # Outputs:
    #   r: melatonin suppression 

    # parameters of the sigmoid function # Tekieh et al. 2020 - Section 2.3.2, after Equation 9
    r_a = 1
    r_b = 0.031 # W/mˆ2
    r_c = 0.82 

    r = 1 - (r_a/(1+power(E_emel/r_b,-r_c))) # Tekieh et al. 2020 - Equation 9
    return r

melatonin_suppression_v = vectorize(melatonin_suppression)

def model(y, t, input_function, forced_wake, minE, maxE, version = '2020'):
    V_v, V_m, H, X, Y, P, Theta_L = y

    IE    = input_function(t)
    S     = state(V_m) 
    # so many things can go wrong with this sigmoid definition
    # what's the threshold irradiance that creates a locally measurable impact on the KSS?
    Sigmoid = ( sigmoid(IE, version) - sigmoid(minE, version) ) / ( sigmoid(maxE, version) - sigmoid(minE, version) ) # Tekieh et al. 2020 - Section 2.3.3: scaling to [0,1]
    alpha = photoreceptor_conversion_rate(IE, S, version)
    Q_m   = mean_population_firing_rate(V_m)
    Q_v   = mean_population_firing_rate(V_v)
    C     = circadian_drive(X,Y)
    D_v   = total_sleep_drive(H,C)
    D_n   = nonphotic_drive(X, S)
    D_p   = photic_drive(X, Y, P, alpha)
    F_w   = forced_wake(t)
    W     = wake_effort(Q_v, F_w)
    
    gradient_y = [(nu_vm*Q_m - V_v + D_v)/tau_v, # V_v, Postnova et al. 2018 - Table 1, Equation 1
                  (nu_mv*Q_v - V_m + D_m + W)/tau_m, # V_m, Postnova et al. 2018 - Table 1, Equation 2
                  (nu_Hm*Q_m - H)/tau_H, # H, Postnova et al. 2018 - Table 1, Equation 3
                  (Y + gamma*(X/3.0 + power(X,3)*4.0/3.0 - power(X,7)*256.0/105.0) + nu_Xp*D_p + nu_Xn*D_n)/tau_X, # X, Postnova et al. 2018 - Table 1, Equation 4
                  (D_p*(nu_YY*Y - nu_YX*X) - power((delta/tau_C),2)*X)/tau_Y, # Y, Postnova et al. 2018 - Table 1, Equation 5
                  alpha*(1-P)-(beta*P), # P, Postnova et al. 2018 - Table 1, Equation 6, revised
                  (-Theta_L + nu_LA*Sigmoid)/tau_L # Tekieh et al. 2020 - Equation 13
                 ] 
    return gradient_y