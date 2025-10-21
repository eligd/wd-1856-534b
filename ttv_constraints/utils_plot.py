import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
plt.rcParams['font.family'] = 'serif'

# for 2d density plot
import seaborn as sns
from scipy.stats import gaussian_kde

import corner

def plot_transit_times(t_obs, epochs_obs, t_pred, epochs_pred, uncertainty, savepath):
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.2)
    fontsize = 14
    leg_size = 13

    for ax in axs:
        ax.minorticks_on()
        ax.tick_params(axis='both', labelsize=12)
        ax.tick_params(direction='in', which='minor', length=5, bottom=True, top=True, left=True, right=True)
        ax.tick_params(direction='in', which='major', length=10, bottom=True, top=True, left=True, right=True)

    def linear_model(x, m, b):
        return m * x + b

    popt_obs, _ = curve_fit(linear_model, epochs_obs, t_obs, sigma=uncertainty, absolute_sigma=True)

    m_obs, b_obs = popt_obs

    t_const_P_obs = m_obs * epochs_obs + b_obs

    tt = []
    for epoch in epochs_obs:
        idx = np.where(epochs_pred == epoch)[0]
        if idx.size != 0:
            tt.append(t_pred[idx[0]])
    tt = np.array(tt)

    axs[0].errorbar(epochs_obs[0:-20], (t_obs[0:-20]-t_const_P_obs[0:-20])*24*60, color='#dbb6b6', yerr=uncertainty[0:-20]*24*60, linestyle='', marker='.', markersize=10, label='Observed (Past Works)', zorder=2)
    axs[1].errorbar(epochs_obs[0:-20], (t_obs[0:-20]-tt[0:-20])*24*60, color='lightsteelblue', yerr=uncertainty[0:-20]*24*60, linestyle='', marker='.', markersize=10, label='Observed (Past Works)', zorder=2)

    axs[0].errorbar(epochs_obs[-20:], (t_obs[-20:]-t_const_P_obs[-20:])*24*60, color='darkred', yerr=uncertainty[-20:]*24*60, linestyle='', marker='.', markersize=10, label='Observed (This Work)', zorder=2)
    axs[1].errorbar(epochs_obs[-20:], (t_obs[-20:]-tt[-20:])*24*60, color='darkblue', yerr=uncertainty[-20:]*24*60, linestyle='', marker='.', markersize=10, label='Observed (This Work)', zorder=2)

    labels = ['Constant Period', 'Companion Model']
    for ax, label in zip(axs, labels):
        xmin, xmax = ax.get_xlim()
        ax.plot([xmin, xmax], [0, 0], color='black', linewidth=2, label=label, zorder=1)
        ax.set_ylim(-0.5, 0.5)

        ax.set_xlabel('Epoch Number', fontsize=fontsize)
        ax.set_ylabel('Transit Timing Variation [min]', fontsize=fontsize)

        ax.legend(loc='upper right', fontsize=leg_size, framealpha=1)

    plt.savefig(savepath, format='pdf', bbox_inches='tight');

def plot_likelihood_ratio(chi2_arr, savepath, const_P_chi2=103.29013566804598, title=None, log=False, eccentric=False):
    fig, ax = plt.subplots(figsize=(15, 8))
    fontsize = 20
    title_size = 25

    #const_P_chi2 = 92.56439191559153
    likelihood_ratio_log_base_e = - (chi2_arr - const_P_chi2) / 2
    likelihood_ratio_log_base_10 = likelihood_ratio_log_base_e * np.log10(np.e)
    if eccentric:
        likelihood_ratio = np.max(likelihood_ratio_log_base_10, axis=(-2,-1)) # collapse along mean anomaly and argument axes
    else: 
        likelihood_ratio = np.max(likelihood_ratio_log_base_10, axis=-1) # collapse along mean anomaly axis
    print("Max likelihood ratio: ", np.max(10**likelihood_ratio))
    print("Max likelihood ratio idx: ", np.argmin(chi2_arr))
    print("Min chi2: ", np.min(chi2_arr))

    print(likelihood_ratio.T)

    if log:
        ax.set_yscale('log')
        im = ax.imshow(likelihood_ratio.T, extent=[0.000003*1047.57, 0.0124*1047.57, 1500, 50], vmin=-4, vmax=4)
    else: 
        im = ax.imshow(likelihood_ratio.T, extent=[0.000003*1047.57, 0.0124*1047.57, 1500, 50], vmin=-4, vmax=4)

    #im = ax.imshow(likelihood_ratio.T, extent=[0.000003*1047.57, 0.0124*1047.57, 1500, 50], vmin=-4, vmax=4)
    #im = ax.imshow(likelihood_ratio.T, extent=[0.000003*1047.57, 0.00381835*1047.57, 400, 50], vmin=-4, vmax=4)
    ax.set_aspect('auto')
    ax.set_xlabel('Mass [Jupiter masses]', fontsize=fontsize)
    ax.set_ylabel('Period [days]', fontsize=fontsize)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Log Base 10 Likelihood Ratio', rotation=90, labelpad=15, fontsize=18)

    dashed_box = Rectangle((0.000003*1047.57, 50), (0.00381835-0.000003)*1047.57, 350, fill=False, edgecolor='black', linestyle='--', linewidth=2)
    ax.add_patch(dashed_box)

    if title is not None:
        plt.title(title, fontsize=title_size, y=1.02)

    plt.savefig(savepath, format='pdf', bbox_inches='tight')

def plot_delta_bic(chi2_arr, savepath, const_P_chi2=103.44948907084853, title=None):
    fig, ax = plt.subplots(figsize=(15, 8))
    fontsize = 20
    title_size = 25

    # companion planet model
    k_companion = 5 # p1 period + p1 reference transit + p2 period + p2 mass + p2 mean anomaly
    n = 68 # number of transit observations
    likelihood_companion = np.exp(-chi2_arr / 2)
    likelihood_companion = np.max(likelihood_companion, axis=-1) # collapse along mean anomaly axis
    bic_companion = k_companion * np.log(n) - 2 * np.log(likelihood_companion)

    # constant period model
    k_const = 2 # p1 period + p1 reference transit
    likelihood_const = np.exp(-const_P_chi2 / 2)
    bic_const = k_const * np.log(n) - 2 * np.log(likelihood_const)

    delta_bic = bic_const - bic_companion

    cmap = cm.viridis.copy()
    cmap.set_over('cyan')
    ax.set_yscale('log')
    im = ax.imshow(delta_bic.T, extent=[0.000003*1047.57, 0.0124*1047.57, 1500, 50], vmin=-100, vmax=-5, cmap=cmap)
    print("max ∆BIC: ", np.max(delta_bic))
    ax.plot([2.3, 2.3], [1000, 100])
    ax.plot([0, 14], [700, 700])

    #im = ax.imshow(delta_bic.T, extent=[0.000003*1047.57, 0.00381835*1047.57, 400, 50], vmin=-100, vmax=-5)

    ax.set_aspect('auto')
    ax.set_xlabel('Mass [Jupiter masses]', fontsize=fontsize)
    ax.set_ylabel('Period [days]', fontsize=fontsize)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('$\Delta$ BIC', rotation=90, labelpad=15, fontsize=18)

    #dashed_box = Rectangle((0.000003*1047.57, 50), (0.00381835-0.000003)*1047.57, 350, fill=False, edgecolor='black', linestyle='--', linewidth=2)
    #ax.add_patch(dashed_box)

    if title is not None:
        plt.title(title, fontsize=title_size, y=1.02)

    plt.savefig(savepath, format='pdf', bbox_inches='tight')

def trace_plots(sampler, savepath):
    fig, axes = plt.subplots(8, figsize=(8,8), sharex=True)
    samples = sampler.get_chain()
    labels = ["Mass", "Period", "Eccentricity", "Inclination", "Longnode", \
              "Argument", "Mean Anomaly", "Log(f)"]
    for i in range(len(labels)):
        ax = axes[i]
        ax.plot(samples[:,:,i],"k",alpha=0.3)
        ax.set_xlim(0,len(samples))
        ax.set_ylabel(labels[i], fontsize=8)
        ax.yaxis.set_label_coords(-0.1,0.5)
        
    axes[-1].set_xlabel("step number")
    plt.savefig(savepath, format='pdf', bbox_inches='tight')

def corner_plots(sampler, theta_ml, savepath):
    flat_samples = sampler.get_chain(discard=2_500, flat=True)
    labels = ["Mass [Jupiter masses]", "Period [days]", "Eccentricity", "Inclination", "LAN", \
              "Arg. periastron", "Mean anomaly", "Log(f)"]

    fig = corner.corner(flat_samples, labels=labels, truths = theta_ml)
    plt.savefig(savepath, format='pdf', bbox_inches='tight')

def plot_2d_density(flat_samples, savepath):
    masses = flat_samples[:,0]
    periods = flat_samples[:,1]
    data = np.vstack([masses, periods])

    kde = gaussian_kde(data)
    density = kde(data)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(masses, periods, s=4, c=density, cmap='magma', alpha=0.5)

    xx, yy = np.mgrid[masses.min():masses.max():200j, periods.min():periods.max():200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde(positions)
    zz = zz.reshape(xx.shape)
    cs = ax.contour(xx, yy, zz, levels=[zz.max()*0.5, zz.max()*0.6, zz.max()*0.7, zz.max()*0.8, zz.max()*0.9], colors='black')

    ax.set_xlabel('Mass [Jupiter masses]', fontsize=20)
    ax.set_ylabel('Period [days]', fontsize=20)
    ax.set_xlim(-0.0004*1047.57, 0.013*1047.57)
    ax.set_ylim(0, 1550)

    ax.minorticks_on()
    ax.tick_params(axis='both', labelsize=12)
    ax.tick_params(direction='in', which='minor', length=5, bottom=True, top=True, left=True, right=True)
    ax.tick_params(direction='in', which='major', length=10, bottom=True, top=True, left=True, right=True)

    plt.savefig(savepath, format='pdf', bbox_inches='tight')

if __name__ == '__main__':
    chi2_arr = np.load('arrays/chi2_new_log.npy') #chi2_zoom_log.npy')
    savepath = 'plots/likelihood_ratios_new_log.pdf' #zoom_log.pdf'
    plot_likelihood_ratio(chi2_arr, savepath, log=True, eccentric=False)

    savepath = 'plots/delta_bic.pdf'
    plot_delta_bic(chi2_arr, savepath)

    #flat_samples = np.load('arrays/flat_samples.npy')
    #flat_samples[:,0] *= 1047.57
    #plot_2d_density(flat_samples, 'mcmc_plots/mcmc_density.pdf')