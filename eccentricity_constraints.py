import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'

# read in data
df = pd.read_csv('all_times.csv')
t = df['T_mid'].to_numpy()
epoch = df['Orbit number'].to_numpy()
uncertainty = df['Uncertainty (days)'].to_numpy()
t0_true = 2458779.3750830120
P_true = 1.4079392114 

# construct grid in eccentricity and omega-dot
eccentricity_arr = np.logspace(-3, 0, 1_000)
omega_dot_arr = np.linspace(-4e-6, 4e-6, 1_000) #(-5e-8, 5e-8, 1_000)
# given index idx between 0 and 999999, take eccentricity_idx = idx // 1000 and omega_dot_idx = idx % 1000

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
omega0 = np.pi/2
Pa = P_true
t0 = t0_true
chi2_arr = np.zeros((1000, 1000))
I_modeled_best = 99999
chi2_best = 99999
for idx in range(1_000_000):
    eccentricity_idx = idx // 1000
    omega_dot_idx = idx % 1000

    domega_dE = omega_dot_arr[omega_dot_idx]
    Ps = Pa * (1 - domega_dE / (2*np.pi))
    e = eccentricity_arr[eccentricity_idx]
    omega = omega0 + epoch * domega_dE
    I_modeled = t0 + Ps * epoch - e * Pa * np.cos(omega) / np.pi
    chisq = chi_squared_reduced(I_data, I_modeled, I_sigma)
    chi2_arr[eccentricity_idx, omega_dot_idx] = chisq
    if chisq < chi2_best:
        I_modeled_best = I_modeled
        chi2_best = chisq
chi2_arr = np.array(chi2_arr)

# ∆BIC plot
k_const = 2 # t0 + period
n = len(t)
bic_const = k_const * np.log(n) + chisq_const

k_companion = 4 # t0 + period + e + omega-dot
bic_companion = k_companion * np.log(n) + chi2_arr

delta_bic = bic_companion - bic_const
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# cmap = ax.imshow(delta_bic, extent=[omega_dot_arr[0], omega_dot_arr[-1], eccentricity_arr[0], eccentricity_arr[-1]], aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=10)
eccentricity, omegadot = np.meshgrid(eccentricity_arr, omega_dot_arr, indexing='ij')
cmap = ax.pcolormesh(omegadot*1e6, eccentricity, delta_bic, shading='auto', cmap='viridis_r', vmin=7, vmax=10)

cbar = plt.colorbar(cmap, ax=ax, pad=0.02)
cbar.set_label('$\Delta$ BIC', rotation=90, labelpad=3, fontsize=24)
cbar.ax.tick_params(labelsize=14)

ax.set_yscale('log')
ax.set_xlabel(r'$\frac{d\omega}{dE} \;[10^{-6}\; \rm rad \; epoch^{-1}]$', fontsize=24)
ax.set_ylabel(r'Eccentricity', fontsize=24)
ax.tick_params(axis='both', which='major', direction='in', labelsize=14, length=10)
ax.tick_params(axis='both', which='minor', direction='in', length=5)
savepath = 'imgs/chi2_arr_eccentricity.pdf'
cmap.set_rasterized(True)
plt.savefig(savepath, format='pdf', bbox_inches='tight');

# best model plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
#plt.plot(epoch, t, color='black')
plt.plot(epoch, t0_true + epoch*P_true - t, color='blue')
plt.plot(epoch, I_modeled_best - t, color='orange')
savepath = 'imgs/test.png'
cmap.set_rasterized(True)
plt.savefig(savepath, format='png', bbox_inches='tight');

mask = delta_bic < 10
allowed_eccentricities = eccentricity_arr[np.any(mask, axis=1)]
print("Max eccentricity: ", np.max(allowed_eccentricities))

allowed_omega_dot = omega_dot_arr[np.any(mask, axis=0)]
print("Max omega dot: ", np.max(np.abs(allowed_omega_dot)))