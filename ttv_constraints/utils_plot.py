import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'

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

    axs[0].plot(epochs_obs, (t_obs-t_const_P_obs)*24*60, color='cornflowerblue', linestyle='', marker='.', markersize=10, label='Observed', zorder=2)
    axs[1].plot(epochs_obs, (t_obs-tt)*24*60, color='darkred', linestyle='', marker='.', markersize=10, label='Observed', zorder=2)

    labels = ['Constant Period', 'Companion Model']
    for ax, label in zip(axs, labels):
        xmin, xmax = ax.get_xlim()
        ax.plot([xmin, xmax], [0, 0], color='black', linewidth=2, label=label, zorder=1)
        ax.set_ylim(-0.5, 0.5)

        ax.set_xlabel('Epoch Number', fontsize=fontsize)
        ax.set_ylabel('Transit Timing Variation [min]', fontsize=fontsize)

        ax.legend(loc='upper right', fontsize=leg_size, framealpha=1)

    plt.savefig(savepath, format='pdf', bbox_inches='tight');

def plot_likelihood_ratio(chi2_arr, savepath, title=None):
    fig, ax = plt.subplots(figsize=(15, 8))
    fontsize = 20
    title_size = 25

    const_P_chi2 = 9049.57294572682 #91.7350202373991
    likelihood_ratio_log_base_e = - (chi2_arr - const_P_chi2) / 2
    likelihood_ratio_log_base_10 = likelihood_ratio_log_base_e * np.log10(np.e)
    likelihood_ratio = np.max(likelihood_ratio_log_base_10, axis=-1) # collapse along mean anomaly axis
    print("Max likelihood ratio: ", np.max(10**likelihood_ratio))
    print("Max likelihood ratio idx: ", np.argmin(chi2_arr))
    print("Min chi2: ", np.min(chi2_arr))

    im = ax.imshow(likelihood_ratio.T, extent=[0.000003*1047.57, 0.0124*1047.57, 1500, 50], vmin=-4, vmax=4)
    ax.set_aspect('auto')
    ax.set_xlabel('Mass [Jupiter masses]', fontsize=fontsize)
    ax.set_ylabel('Period [days]', fontsize=fontsize)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Log Base 10 Likelihood Ratio', rotation=90, labelpad=15, fontsize=18)

    if title is not None:
        plt.title(title, fontsize=title_size, y=1.02)
    
    plt.savefig(savepath, format='pdf', bbox_inches='tight')

if __name__ == '__main__':
    chi2_arr = np.load('arrays/chi2_new.npy')
    savepath = 'plots/likelihood_ratios_new.pdf'
    plot_likelihood_ratio(chi2_arr, savepath)