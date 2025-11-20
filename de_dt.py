import numpy as np

G = 6.674e-11

def de_dt(k_p, Q_p, M_s, n, R_p, e, a):
    return 21/2 * k_p / Q_p * G * M_s**2 * n * R_p**5 * e**2 / a**6

e = 3e-6
Q_p = 3.1e6
M_s = 1.03e30
R_p = 8.4 * 0.013 * 6.957e8
a = 1/0.00294 * 0.013 * 6.957e8

R_p = 0.928 * 7.149e7
a = 0.0204 * 1.496e11
n = np.sqrt(G*M_s/a**3)
omega_dot = 2.8e-8 / (2*np.pi)
M_p = 5.2 * 1.898e27
k_p = omega_dot / (15 * np.pi * M_s / M_p * (R_p / a)**5)

print("k_p: ", k_p)
print("dE/dt: ", de_dt(k_p, Q_p, M_s, n, R_p, e, a))