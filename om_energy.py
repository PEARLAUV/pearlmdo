#%%
from utils import quickModel
import autograd.numpy as np
import openmdao.api as om

m = quickModel()

m.set_var('P_hotel', 50.0)
m.set_var('eta_s', 0.27)
m.set_var('phi_s', 800)
m.set_var('theta_bar', 55*np.pi/180)
m.set_var('I_d', 0.9)
m.set_var('D', 0.005)
m.set_var('L_s', 10)
m.set_var('t_r', 12*3600)
m.set_var('E_AUV', 1900*3600)
m.set_var('t_service', 12*3600)
m.set_var('gamma', 2)
#m.set_var('A_s', 4)
m.set_var('t_mission', 24*3600)

m.add_eq('P_recharge', lambda eta_s, phi_s, theta_bar, I_d, L_s, A_s, D:(   
 eta_s * phi_s * np.cos(theta_bar) * I_d * (1-D)**L_s * A_s))
m.add_eq('P_drawrecharge', lambda P_hotel, P_recharge: P_hotel - P_recharge)
m.add_eq('E_recharge_gen', lambda P_recharge, t_r: P_recharge * t_r)
m.add_eq('E_service', lambda E_AUV, gamma: E_AUV * gamma)
m.add_eq('P_service', lambda E_service, t_service: E_service / t_service)
m.add_eq('P_drawservice', lambda P_hotel, P_service: P_hotel - P_service)


# Propulsion
m.set_var('rho', 1023.6) # density of seawater [kg/m^3]
m.set_var('C_d', 1) # estimate drag coefficient (a square flat plate at 90 deg to the flow is 1.17)
#m.set_var('S_w', 0.5) # will technically need to determine from A_solar and other structural needs... assume something for now
m.set_var('V', 0.3) # [m/s]
m.set_var('eta_m', 0.75) # estimated, need to determine from motors?
m.add_eq('P_move', lambda rho, C_d, A_s, V, eta_m: rho*C_d*0.1*A_s*V**3/(2*eta_m))

# Balances 
m.add_eq('E_required', lambda P_hotel, t_mission, E_service, P_move: (
     E_service + (P_hotel + P_move)* t_mission))

m.set_equal('E_recharge_gen', 'E_required', solvefor='A_s')

# Battery
m.set_var('mu_batt', 30*3600)
m.set_var('DOD', 0.4)
m.set_var('eta_trans', 0.85)
m.set_var('nu_batt', 450*1000*3600)
m.set_var('N', 1)
m.add_eq('C', lambda E_AUV, P_service, t_service, DOD, N, eta_trans: (
      E_AUV + P_service*t_service)/(DOD*N*eta_trans))
m.add_eq('m_batt', lambda C, mu_batt: C/mu_batt + 5)
m.add_eq('V_batt', lambda C, nu_batt: C/nu_batt)


prob = m.prob
prob.setup()
#prob.run_model()
m.run()
#prob.check_partials(compact_print=True)

# model = prob.model
# model.add_design_var('S', lower=0, upper=80)
# model.add_design_var('V', lower=0, upper=50)
# model.add_objective('D')
# prob.driver = om.DOEDriver(om.UniformGenerator(num_samples=20))
# prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))
#m.run()

# prob.setup()
# prob.run_driver()
# prob.cleanup()

# cr = om.CaseReader("cases.sql")
# cases = cr.list_cases('driver')

# print(len(cases))

# values = []
# for case in cases:
#     outputs = cr.get_case(case).outputs
#     values.append((outputs['S'], outputs['V'], outputs['D']))

# print("\n".join(["S: %5.2f, V: %5.2f, D: %6.2f" % xyf for xyf in values]))

# %%

