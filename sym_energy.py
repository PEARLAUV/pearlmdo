from sympywrap import var_generator
import sympy as sp
import numpy as np
from utils import quickModel
import openmdao.api as om

m = quickModel()
var = var_generator(m)

P_hotel = var('P_hotel', 50, 'W')
eta_s = var('eta_s', 0.27)
phi_s = var('phi_s', 800, 'W/m**2')
theta_bar = var('theta_bar', 55, 'deg')
I_d = var('I_d', 0.9)
D = var('D', 0.005)
L_s = var('L_s', 10)
t_r = var('t_r', unit='hr')
E_AUV = var('E_AUV', 1.9, 'kW*hr')
t_service = var('t_service', 12, 'hr')
gamma = var('gamma', 2)
A_s = var('A_s', 4, 'm**2')
t_mission = var('t_mission', 24, 'hr')

# AUV recharging system
P_recharge = var('P_recharge', 
    eta_s * phi_s * sp.cos(theta_bar) * I_d * (1-D)**L_s * A_s, 'W')
P_drawrecharge = var('P_drawrecharge', P_hotel - P_recharge, 'W')
E_recharge_gen = var('E_recharge_gen', P_recharge * t_r, 'W*hr')
E_service = var('E_service', E_AUV * gamma, 'W*hr')
P_service = var('P_service', E_service / t_service, 'W')
P_drawservice = var('P_drawservice', P_hotel - P_service, 'W')

# Propulsion
rho = var('rho', 1023.6, 'kg/m**3') # density of seawater [kg/m^3]
C_d = var('C_d', 1) # estimate drag coefficient (a square flat plate at 90 deg to the flow is 1.17)
S_w = var('S_w', 0.5, 'm**2') # will technically need to determine from A_solar and other structural needs... assume something for now
V = var('V', 0.3, 'm/s') # [m/s]
eta_m = var('eta_m', 0.75) # estimated, need to determine from motors?
P_move = var('P_move', rho*C_d*S_w*V**3/(2*eta_m), 'W')

# Balances
E_required = var('E_required', E_service + (P_hotel + P_move)* t_mission, 'W*hr')

m.set_equal('E_recharge_gen', 'E_required', solvefor='t_r')

# Battery
mu_batt = var('mu_batt', 30, 'W*hr/kg')
DOD = var('DOD', 0.4)
eta_trans = var('eta_trans', 0.85)
nu_batt = var('nu_batt', 450, 'kW*hr/(m**3)')
N = var('N', 1)
C = var('C', E_service/(DOD*N*eta_trans), 'kW*hr')
m_batt_zero = var('m_batt_zero', 5, 'kg')
m_batt = var('m_batt', C/mu_batt + m_batt_zero, 'kg')
V_batt = var('V_batt', C/nu_batt, 'm**3')

m.run()