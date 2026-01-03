import yaml
import numpy as np
import pandas as pd
import argparse
import ttvfast

from utils import *
from utils_plot import *


def get_epoch(midtransit):
    period = p1_period
    epoch = np.around((midtransit - ref_transit) / period)
    return epoch

def get_transit_time_predictions(theta):
    p2_mass, p2_period, p2_eccentricity, p2_inclination, p2_longnode, \
    p2_argument, p2_mean_anomaly = theta
    planet2 = ttvfast.models.Planet(
        mass = p2_mass,
        period = p2_period,
        eccentricity = p2_eccentricity,
        inclination = p2_inclination,
        longnode = p2_longnode,
        argument = p2_argument,
        mean_anomaly = p2_mean_anomaly
    )

    planets = [planet1, planet2]
    start_time = ref_transit + start_epoch * p1_period
    end_time = start_time + duration
    results = ttvfast.ttvfast(planets, stellar_mass, start_time, time_step, end_time)
    planet_id = np.array(results['positions'][0])
    transit_times = np.array(results['positions'][2])
    mask = np.logical_and(planet_id == 0, transit_times >= 0)
    return transit_times[mask]

def linear_model(x, m, b):
    return m*x + b


# get config file
parser = argparse.ArgumentParser(description='TTV constraints using grid search')
add_arg = parser.add_argument
add_arg('--config_file', help='configuration file')
args = parser.parse_args()

with open(args.config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# get observed transit times
ref_transit = config['ref_transit']
filepath = config['transit_times_filepath']
df = pd.read_csv(filepath)
t_obs = df['T_mid'].to_numpy()
t_obs_err = df['Uncertainty (days)'].to_numpy()

savepath = config['savepath']

# read physical parameters and integration parameters from config
G = config['G']
stellar_mass = config['stellar_mass']
time_step = config['time_step']
start_epoch = config['start_epoch']
duration = config['duration']

# WD 1856+534 b
p1_mass = config['p1_mass']
p1_period = config['p1_period']
p1_eccentricity = config['p1_eccentricity']
p1_inclination = config['p1_inclination']
p1_longnode = config['p1_longnode']
p1_argument = config['p1_argument']
p1_mean_anomaly = config['p1_mean_anomaly']

planet1 = ttvfast.models.Planet(
    mass = p1_mass,
    period = p1_period,
    eccentricity = p1_eccentricity,
    inclination = p1_inclination,
    longnode = p1_longnode,
    argument = p1_argument,
    mean_anomaly = p1_mean_anomaly
)

epochs_obs = get_epoch(t_obs)

# grid search parameters
p2_eccentricity = config['p2_eccentricity']
p2_inclination = config['p2_inclination']
p2_longnode = config['p2_longnode']
p2_argument = config['p2_argument']

p2_mass_min = config['p2_mass_min']
p2_mass_max = config['p2_mass_max']
p2_mass_points = config['p2_mass_points']
p2_masses = np.linspace(p2_mass_min, p2_mass_max, p2_mass_points)

p2_period_min = config['p2_period_min']
p2_period_max = config['p2_period_max']
p2_period_points = config['p2_period_points']
p2_periods = np.linspace(p2_period_min, p2_period_max, p2_period_points)

p2_mean_anomaly_min = config['p2_mean_anomaly_min']
p2_mean_anomaly_max = config['p2_mean_anomaly_max']
p2_mean_anomaly_points = config['p2_mean_anomaly_points']
p2_mean_anomalies = np.linspace(p2_mean_anomaly_min, p2_mean_anomaly_max, p2_mean_anomaly_points)

popt, _ = curve_fit(linear_model, epochs_obs, t_obs, sigma=t_obs_err, absolute_sigma=True)
m, b = popt
obs_ttv = t_obs - (m * epochs_obs + b)


if __name__ == '__main__':
    chi2_arr = np.load(savepath)
    idx = np.argmin(chi2_arr)
    grid_params = np.meshgrid(p2_masses, p2_periods, p2_mean_anomalies, indexing='ij')
    mass_idx = idx // (p2_period_points * p2_mean_anomaly_points)
    period_idx = (idx % (p2_period_points * p2_mean_anomaly_points)) // p2_mean_anomaly_points
    mean_anomaly_idx = idx % p2_mean_anomaly_points

    p2_mass = p2_masses[mass_idx]
    p2_period = p2_periods[period_idx]
    p2_mean_anomaly = p2_mean_anomalies[mean_anomaly_idx]

    theta = [p2_masses[mass_idx], p2_periods[period_idx], p2_eccentricity, p2_inclination, 
             p2_longnode, p2_argument, p2_mean_anomalies[mean_anomaly_idx]]
    t_pred = get_transit_time_predictions(theta)
    epochs_pred = get_epoch(t_pred)

    t_p = []
    for epoch in epochs_obs:
        idx = np.where(epochs_pred == epoch)[0]
        if idx.size != 0:
            t_p.append(t_pred[idx[0]])
    t_p = np.array(t_p)
    R = romer_delay(p1_period, p2_periods[period_idx], p1_mass, p2_masses[mass_idx], stellar_mass, p1_inclination, ref_transit, t_p)
    t_p += R

    ref_idx = np.where(epochs_obs==get_epoch(ref_transit))[0][0]
    offset = t_obs[ref_idx] - t_p[ref_idx]
    t_p += offset

    # for residual panel
    t_p_copy = np.array(t_pred)
    R = romer_delay(p1_period, p2_periods[period_idx], p1_mass, p2_masses[mass_idx], stellar_mass, p1_inclination, ref_transit, t_p_copy)
    t_p_copy += R

    ref_idx = np.where(epochs_obs==get_epoch(ref_transit))[0][0]
    offset = t_obs[ref_idx] - t_p_copy[ref_idx]
    t_p_copy += offset

    savepath = 'plots/o-c_new_zoom_res.pdf'
    print("Mass: ", p2_masses[mass_idx])
    print("Period: ", p2_periods[period_idx])
    print("Mean anomaly: ", p2_mean_anomalies[mean_anomaly_idx])
    print("Min chi2: ", np.min(chi2_arr))
    plot_transit_times_new(t_obs, epochs_obs, t_p, get_epoch(t_p), t_p_copy, get_epoch(t_p_copy), t_obs_err, savepath)