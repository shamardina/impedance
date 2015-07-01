# -*- coding: utf-8 -*-
"""
Constants, parameters and experimental conditions of the fuel cell and
experiment. Auxiliary functions for dependent parameters.
"""

eff_zero = 1e-16
eff_inf  = 1e+16

R = 8.31                           # J/mol/K
T_zero = 273.15                    # K
T = T_zero + 160.0                 # K
F = 96485.0                        # C/mol
p = 101325.0                       # Pa

l_t = 0.3e-4                       # m
l_d = 3.8e-4                       # m
l_m = 6.0e-5                       # m
h = 1.0e-3                         # m

D_O = 4.1e-5                       # m^2/s
sigma = 222.0                      # S/m

gas_O = 0.21
c_tot = p/R/T                      # mol/m^3
c_ref = c_tot*gas_O                # mol/m^3

# Initial parameters for fitting:
j_0 = 0.6                          # A/m^2
b = 0.08                           # V
Cdl = 1000.0/l_t                   # F/m^3
sigma_t = 5.0                      # S/m
epsilon_GDL_gas = 0.7
D_O_GDL = D_O*epsilon_GDL_gas**1.5 # m^2/s
D_O_CCL = D_O_GDL # m^2/s

def alpha(b):
    """Transfer coefficient of the oxygen reduction reaction."""
    return R*T/b/F

def kappa(R_Ohm):
    """Returns membrane conductivity [S/m]."""
    return l_m/(R_Ohm - 2*l_d/sigma) # S/m

def flow(J, lam, area_cm, scale=10000.0):
    """Returns oxygen inlet flow velocity in mNl/min."""
    return lam/(4.0*F/(J*area_cm/scale))*60.0/((1.0e-3*p/T_zero/R*gas_O*1.0e-3))

def stoich(J, flow, area_cm, scale=10000.0):
    """Returns oxygen stoichimetry lambda. flow is in mNl/min."""
    return 4.0*F/(J*area_cm/scale)*(p/T_zero/R*gas_O*flow*1.0e-3*1.0e-3)/60.0

def current(lam, flow, area_cm, scale=10000.0):
    """Returns current density at fixed flow velocity and stoichiometry [A/m^2]."""
    return 4.0*F/(area_cm/scale)*(flow*1.0e-3*p/T_zero/R*gas_O*1.0e-3)/60.0/lam # A/m^2
