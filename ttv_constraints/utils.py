import numpy as np

def romer_delay(P_inner, P_outer, M_inner, M_outer, M_star, i, t0, t):
    # all calculations in units of AU, days, M_sun, radians
    G = 0.000295994511
    c = 173.145

    u = 2*np.pi / P_inner * (t-t0)
    A = G**(1/3) / (c * (2*np.pi)**(2/3)) * P_outer**(2/3) * M_outer * np.sin(i*np.pi/180) / (M_star + M_inner + M_outer)**(2/3)
    R = A * np.sin(u)
    return R