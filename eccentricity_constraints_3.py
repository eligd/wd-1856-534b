import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.rcParams['font.family'] = 'serif'

# read in data
df = pd.read_csv('all_times.csv')
t = df['T_mid'].to_numpy()
epoch = df['Orbit number'].to_numpy()
uncertainty = df['Uncertainty (days)'].to_numpy()
t0_true = 2458779.3750830120
P_true = 1.4079392114 

# construct grid in eccentricity and omega0
k_p = 0.565 # for Jupiter
domega_dE = 15 * np.pi * k_p * (0.518 * 1.989e30 / (5.2 * 1.898e27)) * (8.4*0.00294)**5
print(domega_dE)
eccentricity_arr = np.logspace(-14, 0, 2000)
omega0_arr = np.linspace(-np.pi, np.pi, 2000)
# given index idx between 0 and 999999, take eccentricity_idx = idx // 2000 and omega0_idx = idx % 2000

# evaluate chi2 for e=0, omgea-dot=0 model
def chi_squared_reduced(data, model, sigma, dof=None):
    """
    Calculate the reduced chi-squared value for a fit.

    If no dof is given, returns the chi-squared (non-reduced) value.

    Parameters
    ----------
    data : array_like
        The observed data.
    model : array_like
        The model data.
    sigma : array_like
        The uncertainty in the data.
    dof : int
        Degrees of freedom (len(data) - # of free parameters).
    """

    sq_residual = (data - model)**2
    chi_sq = np.sum(sq_residual / sigma**2)
    if dof is None:
        return chi_sq
    else:
        nu = len(data) - dof
        return chi_sq / nu

# Define our data, model, uncertainty, and degrees of freedom
I_data = t  # observed data
I_modeled = t0_true + epoch*P_true # model fitted result
I_sigma = uncertainty # uncertainty in the data

# Calculate the Chi-Squared value (no dof)
chisq_const = chi_squared_reduced(I_data, I_modeled, I_sigma)
print(f"e=0 chi-squared test statistic = {chisq_const:1.2f}")

# evaluate chi2 for each point in grid
Pa = P_true
t0 = t0_true
chi2_arr = np.zeros((2000, 2000))
I_modeled_best = 999999999
chi2_best = 999999999
for idx in tqdm(range(2000**2)):
    eccentricity_idx = idx // 2000
    omega0_idx = idx % 2000

    omega0 = omega0_arr[omega0_idx]
    Ps = Pa * (1 - domega_dE / (2*np.pi))
    e = eccentricity_arr[eccentricity_idx]
    omega = omega0 + epoch * domega_dE
    I_modeled = t0 + Ps * epoch - e * Pa * np.cos(omega) / np.pi
    chisq = chi_squared_reduced(I_data, I_modeled, I_sigma)
    chi2_arr[eccentricity_idx, omega0_idx] = chisq
    if chisq < chi2_best:
        I_modeled_best = I_modeled
        chi2_best = chisq
chi2_arr = np.array(chi2_arr)

# ∆BIC plot
k_const = 2 # t0 + period
n = len(t)
bic_const = k_const * np.log(n) + chisq_const

k_companion = 4 # t0 + period + e + omega0
bic_companion = k_companion * np.log(n) + chi2_arr

delta_bic = bic_companion - bic_const
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
eccentricity, omega = np.meshgrid(eccentricity_arr, omega0_arr, indexing='ij')
cmap = ax.pcolormesh(omega, eccentricity, delta_bic, shading='auto', cmap='viridis_r')#, vmin=8, vmax=15)

cbar = plt.colorbar(cmap, ax=ax, pad=0.02)
cbar.set_label('$\Delta$ BIC', rotation=90, labelpad=3, fontsize=24)
cbar.ax.tick_params(labelsize=14)

ax.set_yscale('log')
ax.set_xlabel(r'$\omega_0\;{\rm [rad]}$', fontsize=24)
ax.set_ylabel(r'Eccentricity', fontsize=24)
ax.tick_params(axis='both', which='major', direction='in', labelsize=14, length=10)
ax.tick_params(axis='both', which='minor', direction='in', length=5)
savepath = 'imgs/eccentricity_constraints.pdf'
cmap.set_rasterized(True)
plt.savefig(savepath, format='pdf', bbox_inches='tight');

# best model plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
plt.plot(epoch, t0_true + epoch*P_true - t, color='blue')
plt.plot(epoch, I_modeled_best - t, color='orange')
savepath = 'imgs/test.pdf'
plt.savefig(savepath, format='pdf', bbox_inches='tight');

mask = delta_bic < 10
allowed_eccentricities = eccentricity_arr[np.any(mask, axis=1)]
print("Max eccentricity: ", np.max(allowed_eccentricities))

allowed_omega_dot = omega0_arr[np.any(mask, axis=0)]
print("Max omega dot: ", np.max(np.abs(allowed_omega_dot)))