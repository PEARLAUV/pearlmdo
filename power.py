#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
arr_rate = np.array([2,3,4])
P_servicer = 50
P_n = P_servicer
E_AUV = 1900
T_d = 12
T_n = 24 - T_d
X_d = 0.9
X_n = 0.8
theta_bar = 55
phi_solar = 800
I_d = 0.9
D = 0.005
L = 10
eta = np.linspace(0.15,0.44,1000)
P_EOL = eta*phi_solar*np.cos(
    np.radians(theta_bar))*I_d*((1-D)**L)

for l in arr_rate: 
    P_charge = E_AUV*l/T_d
    P_d = P_servicer + P_charge
    P_req = ((P_d*T_d/X_d)+(P_n*T_n/X_n))/T_d
    A_solar = P_req/P_EOL
    plt.plot(eta, A_solar)

plt.xlabel(r'Solar Cell Efficiency, $\eta$')
plt.ylabel(r'Required Solar Panel Area, $A_{solar}$ [$\mathrm{m}^2$]')
plt.legend(('2 AUV/day','3 AUV/day','4 AUV/day'))
#%%
