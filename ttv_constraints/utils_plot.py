import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'

def plot_transit_times(t_obs, epochs_obs, t_pred, epochs_pred, period, savepath):
    fig, ax = plt.subplots(figsize=(8, 8))
    fontsize = 20
    leg_size = 14

    ax.minorticks_on()
    ax.tick_params(axis='both', labelsize=12)
    ax.tick_params(direction='in', which='minor', length=5, bottom=True, top=True, left=True, right=True)
    ax.tick_params(direction='in', which='major', length=10, bottom=True, top=True, left=True, right=True)

    t_const_P_obs = t_obs[0] + period * (epochs_obs - epochs_obs[0])
    t_const_P_pred = t_pred[0] + period * (epochs_pred - epochs_pred[0])

    ax.plot(epochs_obs, (t_obs-t_const_P_obs)*24*60, color='tab:blue', linestyle='', marker='.', markersize=10, label='Observed')
    ax.plot(epochs_pred, (t_pred-t_const_P_pred)*24*60, color='tab:red', linestyle='-', label='Predicted')

    xmin, xmax = ax.get_xlim()
    ax.plot([xmin, xmax], [0, 0], color='black', linewidth=2, label='Constant Period')
    ax.set_ylim(-0.5, 0.5)

    ax.set_xlabel('Epoch Number', fontsize=fontsize)
    ax.set_ylabel('Transit Timing Variation [min]', fontsize=fontsize)

    ax.legend(loc='upper right', fontsize=leg_size, framealpha=1)

    plt.savefig(savepath, format='pdf', bbox_inches='tight');

def plot_likelihood_ratio(chi2_arr, savepath):
    fig, ax = plt.subplots(figsize=(10, 8))
    fontsize = 20

    const_P_chi2 = 91.73502023749768
    likelihood_ratio_log_base_e = - (chi2_arr - const_P_chi2) / 2
    likelihood_ratio_log_base_10 = likelihood_ratio_log_base_e * np.log10(np.e)
    likelihood_ratio = np.max(likelihood_ratio_log_base_10, axis=-1) # collapse along mean anomaly axis

    im = ax.imshow(likelihood_ratio.T, extent=[0.000003*1047.57, 0.0124*1047.57, 2000, 100], vmin=-3, vmax=3)
    ax.set_aspect('auto')
    ax.set_xlabel('Mass [Jupiter masses]', fontsize=fontsize)
    ax.set_ylabel('Period [days]', fontsize=fontsize)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Log Base 10 Likelihood Ratio', rotation=90, labelpad=15, fontsize=18)
    
    plt.savefig(savepath, format='pdf', bbox_inches='tight')

if __name__ == '__main__':
    chi2_arr = np.load('arrays/chi2_no_romer_correction.npy')
    savepath = 'plots/likelihood_ratio_no_romer_correction.pdf'
    plot_likelihood_ratio(chi2_arr, savepath)

    chi2_arr = np.load('arrays/chi2_romer_correction.npy')
    savepath = 'plots/likelihood_ratio_romer_correction.pdf'
    plot_likelihood_ratio(chi2_arr, savepath)