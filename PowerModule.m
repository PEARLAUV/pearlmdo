function [solar_result, battery_result] = PowerModule(const,params)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%Given Mission Constants: const = [phi_solar,E_AUV,AUV_arrive_rate,T_day,T_service]
%Tunable Parameters: params = [P_servicer,t_life,eta_solar,nu_battery,mu_battery]
%
%Outputs:
%solar_result = [A_solar, m_solar, Cost_solar, P_EOL]
%battery_result = [V_battery, m_battery, Cost_battery, Capacity_battery]


arr_rate = [2 3 4];
figure
hold on
for k = 1:length(arr_rate);
P_servicer = 50; %In W
P_n = P_servicer;
E_AUV = 1900; %in W-hr for Bluefin-21 (but I think papers say baseline Bluefin-9
T_d = 12; %in hr
T_n = 24 - T_d; %in hr
lambda = arr_rate(k); %AUVs per day (i.e. 0.2 means 1 AUV every 5 days)
P_charge = E_AUV*lambda/T_d;
P_d = P_servicer + P_charge;
X_d = 0.9;
X_n = 0.8;
P_req = ((P_d*T_d/X_d) + (P_n*T_n/X_n))/T_d;
theta_bar = 55; %in deg
phi_solar = 800; %W/m^2
I_d = 0.9;
D = 0.005; %degradation in decimal of percent per year
L = 10; %life in years
eta = linspace(0.15,0.44,1000);
P_EOL = eta*phi_solar*cosd(theta_bar)*I_d*((1-D)^L);
A_solar = P_req*(P_EOL.^-1);

plot(eta,A_solar)
end

xlabel('Solar Cell Efficiency, $\eta$','Interpreter','Latex')
ylabel('Required Solar Panel Area, $A_{solar}$ [$\mathrm{m}^2$]','Interpreter','Latex')
legend({'2 AUV/day','3 AUV/day','4 AUV/day'},'Interpreter','Latex')


E_AUV_vec = linspace(1900,13500,1000);
mu_batt_vec = [30 100 200]; %in W-hr/kg for NiCd, NiMH, Li-ion
DOD_vec = [0.4 0.55 0.7];
figure
hold on
for j = 1:length(mu_batt_vec)
DOD = 0.7;
N = 1;
eta_trans = 0.85
C = (E_AUV_vec + P_n*T_n)/(DOD_vec(j)*N*eta_trans);
mu_batt = mu_batt_vec(j); %in W-hr/kg
m_batt = C/mu_batt + 5;
nu_batt = 450*1000; %in W-hr/(m^3)
V_batt = C/nu_batt;
plot(E_AUV_vec,m_batt)
end
xlabel('Energy Demand per Recharged AUV, $E_{AUV}$ [W-hr]','Interpreter','Latex')
ylabel('Mass of Servicing Platform Batteries, $m_{batt}$ [kg]','Interpreter','Latex')
legend({'NiCd, 30 W-hr/kg','NiMH, 100 W-hr/kg','Li-ion 200 W-hr/kg'},'Interpreter','Latex')