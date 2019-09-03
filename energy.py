#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from power import T_n, P_n
E_AUV_vec = np.linspace(1900,13500,1000)
mu_batt_vec = np.array([30, 100, 200]) #in W-hr/kg for NiCd, NiMH, Li-ion
DOD_vec = np.array([0.4, 0.55, 0.7])
DOD = 0.7
N = 1
eta_trans = 0.85
nu_batt = 450*1000 # W-hr/(m^3)

for mu_batt, d in zip(mu_batt_vec, DOD_vec):
    C = (E_AUV_vec + P_n*T_n)/(d*N*eta_trans)
    m_batt = C/mu_batt + 5
    V_batt = C/nu_batt
    plt.plot(E_AUV_vec,m_batt)

plt.xlabel(r'Energy Demand per Recharged AUV, $E_{AUV}$ [W-hr]')
plt.ylabel(r'Mass of Servicing Platform Batteries, $m_{batt}$ [kg]')
plt.legend(('NiCd, 30 W-hr/kg','NiMH, 100 W-hr/kg','Li-ion 200 W-hr/kg'))

#%%
