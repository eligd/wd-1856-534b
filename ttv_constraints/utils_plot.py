import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
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

    axs[0].errorbar(epochs_obs[-20:], (t_obs[-20:]-t_const_P_obs[-20:])*24*60, color='darkred', yerr=uncertainty[-20:]*24*60, linestyle='', marker='^', markersize=6, label='Observed (This Work)', zorder=2)
    axs[1].errorbar(epochs_obs[-20:], (t_obs[-20:]-tt[-20:])*24*60, color='darkblue', yerr=uncertainty[-20:]*24*60, linestyle='', marker='^', markersize=6, label='Observed (This Work)', zorder=2)

    labels = ['Constant Period', 'Companion Model']
    for ax, label in zip(axs, labels):
        xmin, xmax = ax.get_xlim()
        ax.plot([xmin, xmax], [0, 0], color='black', linewidth=2, label=label, zorder=1)
        ax.set_ylim(-0.5, 0.5)

        ax.set_xlabel('Epoch Number', fontsize=fontsize)
        ax.set_ylabel('Transit Timing Variation [min]', fontsize=fontsize)

        ax.legend(loc='upper right', fontsize=leg_size, framealpha=1)

    plt.savefig(savepath, format='pdf', bbox_inches='tight');

def plot_transit_times_tess(t_obs, epochs_obs, t_pred, epochs_pred, t_3, epochs_3, t_p_3, epochs_p_3, uncertainty, upper_err_3, lower_err_3, savepath):
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.2)
    fontsize = 14
    leg_size = 10.5

    for ax in axs:
        ax.minorticks_on()
        ax.tick_params(axis='both', labelsize=12)
        ax.tick_params(direction='in', which='minor', length=5, bottom=True, top=True, left=True, right=True)
        ax.tick_params(direction='in', which='major', length=10, bottom=True, top=True, left=True, right=True)
        ax.ticklabel_format(style='plain')

    def linear_model(x, m, b):
        return m * x + b

    popt_obs, _ = curve_fit(linear_model, epochs_obs, t_obs, sigma=uncertainty, absolute_sigma=True)

    m_obs, b_obs = popt_obs

    t_const_P_obs = m_obs * epochs_obs + b_obs
    t_const_P_3 = m_obs * epochs_3 + b_obs

    tt = []
    for epoch in epochs_obs:
        idx = np.where(epochs_pred == epoch)[0]
        if idx.size != 0:
            tt.append(t_pred[idx[0]])
    tt = np.array(tt)

    tt_3 = []
    for epoch in epochs_3:
        idx = np.where(epochs_p_3 == epoch)[0]
        if idx.size != 0:
            tt_3.append(t_p_3[idx[0]])
    tt_3 = np.array(tt_3)

    axs[0].errorbar(epochs_obs[0:-20], (t_obs[0:-20]-t_const_P_obs[0:-20])*24*60, color='lightsteelblue', yerr=uncertainty[0:-20]*24*60, linestyle='', marker='.', markersize=10, label='Observed (Past Works)', zorder=2)
    axs[1].errorbar(epochs_obs[0:-20], (t_obs[0:-20]-tt[0:-20])*24*60, color='lightsteelblue', yerr=uncertainty[0:-20]*24*60, linestyle='', marker='.', markersize=10, label='Observed (Past Works)', zorder=2)

    axs[0].errorbar(epochs_obs[-20:], (t_obs[-20:]-t_const_P_obs[-20:])*24*60, color='darkblue', yerr=uncertainty[-20:]*24*60, linestyle='', marker='^', markersize=6, label='Observed (This Work)', zorder=2)
    axs[1].errorbar(epochs_obs[-20:], (t_obs[-20:]-tt[-20:])*24*60, color='darkblue', yerr=uncertainty[-20:]*24*60, linestyle='', marker='^', markersize=6, label='Observed (This Work)', zorder=2)

    axs[0].errorbar(epochs_3, (t_3-t_const_P_3)*24*60, color='#009E73', yerr=[lower_err_3*24*60, upper_err_3*24*60], linestyle='', marker='s', markersize=4, label='TESS (Naponiello 2025)', zorder=1, alpha=0.25)
    axs[1].errorbar(epochs_3, (t_3-tt_3)*24*60, color='#009E73', yerr=[lower_err_3*24*60, upper_err_3*24*60], linestyle='', marker='s', markersize=4, label='TESS (Naponiello 2025)', zorder=1, alpha=0.25)
    
    labels = ['Constant Period', 'Companion Model']
    for ax, label in zip(axs, labels):
        xmin, xmax = ax.get_xlim()
        ax.plot([xmin, xmax], [0, 0], color='black', linewidth=2, label=label, zorder=1)
        ax.set_xlim(-100, 1550)
        ax.set_ylim(-2, 2)
        ax.set_yscale('asinh')
        ax.set_yticks([-1, -0.5, -0.1, 0.1, 0.5, 1])

        formatter = ticker.ScalarFormatter()
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_minor_locator(ticker.NullLocator())

        ax.set_xlabel('Epoch Number', fontsize=fontsize)
        ax.set_ylabel('Transit Timing Variation [min]', fontsize=fontsize)

        ax.legend(loc='upper center', ncol=4, fontsize=leg_size, framealpha=0.8)

        ax.grid(which='both', color='lightgray', linestyle='-', alpha=0.3)

    plt.savefig(savepath, format='pdf', bbox_inches='tight');

def plot_likelihood_ratio(chi2_arr, savepath, const_P_chi2=103.44948907084853, title=None, full=True):
    fig, ax = plt.subplots(figsize=(15, 8))
    fontsize = 20
    title_size = 25

    likelihood_ratio_log_base_e = - (chi2_arr - const_P_chi2) / 2
    likelihood_ratio_log_base_10 = likelihood_ratio_log_base_e * np.log10(np.e)
    likelihood_ratio = np.max(likelihood_ratio_log_base_10, axis=-1) # collapse along mean anomaly axis
    print("Max likelihood ratio: ", np.max(10**likelihood_ratio))
    print("Max likelihood ratio idx: ", np.argmax(likelihood_ratio_log_base_10))
    print("Min chi2 idx: ", np.argmin(chi2_arr)) # should be same as max likelihood ratio idx
    print("Min chi2: ", np.min(chi2_arr))

    if full: # not zoomed, grid search on linear scale
        masses = np.linspace(0.000003*1047.57, 0.0124*1047.57, 300) # 1047.57 converts Solar masses to Jupiter masses
        periods = np.linspace(50, 1500, 100)
        im = ax.pcolormesh(masses, periods, likelihood_ratio.T, shading='gouraud', cmap='viridis', edgecolors='none', vmin=-4, vmax=4)
        dashed_box = Rectangle((0.000003*1047.57, 50), (0.00381835-0.000003)*1047.57, 350, fill=False, edgecolor='black', linestyle='--', linewidth=2)
        ax.add_patch(dashed_box)
    else: # zoomed, grid search on log scale
        masses = np.linspace(0.000003*1047.57, 0.00381835*1047.57, 300)
        periods = np.logspace(np.log10(50), np.log10(400), 100)
        im = ax.pcolormesh(masses, periods, likelihood_ratio.T, shading='gouraud', cmap='viridis', edgecolors='none', vmin=-4, vmax=4)
    
    ax.set_aspect('auto')
    ax.set_xlabel('Mass [Jupiter masses]', fontsize=fontsize)
    ax.set_ylabel('Period [days]', fontsize=fontsize)

    ax.yaxis.set_inverted(True)
    ax.set_yscale('log')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Log Base 10 Likelihood Ratio', rotation=90, labelpad=15, fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    ax.tick_params(axis='both', which='major', bottom=True, top=True, right=True, left=True, direction='in', length=12, width=1, labelsize=14)
    ax.tick_params(axis='both', which='minor', bottom=True, top=True, right=True, left=True, direction='in', length=6, width=1, labelsize=14)

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
    scaled_density = (density - density.min()) / (density.max() - density.min())
    cm = ax.scatter(masses, periods, s=4, c=scaled_density, cmap='magma')
    cm.set_alpha(0.5)

    xx, yy = np.mgrid[masses.min():masses.max():200j, periods.min():periods.max():200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde(positions)
    zz = zz.reshape(xx.shape)
    cs = ax.contour(xx, yy, zz, levels=[zz.max()*0.5, zz.max()*0.6, zz.max()*0.7, zz.max()*0.8, zz.max()*0.9], colors='black')

    norm = plt.Normalize(vmin=scaled_density.min(), vmax=scaled_density.max())
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('magma')), ax=ax, pad=0.02)
    cbar.set_label('Normalized Density', rotation=90, labelpad=7, fontsize=20)
    cbar.ax.tick_params(labelsize=12)

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
    chi2_arr = np.load('arrays/chi2.npy')
    savepath = 'plots/likelihood_ratios.pdf'
    plot_likelihood_ratio(chi2_arr, savepath, full=True)