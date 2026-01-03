import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
plt.rcParams['font.family'] = 'serif'

import emcee
import corner
from scipy.optimize import minimize
from IPython.display import Math

import sys
sys.path.append('../fit_lightcurves/')
import model_transits

data_arr = np.load('/Users/eligendreaudistler/Desktop/Berkeley_S26/wd-1856-534b/arrays/data_arr.npy', allow_pickle=True)

dates = ['2022-04-01', '2022-06-02', '2023-07-31', '2023-08-14', '2024-05-13', 
         '2024-06-06', '2024-06-13', '2024-06-23', '2024-06-30', '2024-07-07',
         '2024-07-14', '2024-07-24', '2024-08-07', '2024-08-14', '2025-05-17',
         '2025-05-24', '2025-06-07', '2025-06-17', '2025-07-01']
base_dir = '/Users/eligendreaudistler/Desktop/wd-1856-534b/arrays'

for i, date in enumerate(dates):
    theta_mcmc = np.load('../arrays/theta_mcmc_{}.npy'.format(date))
    data_arr[i]['BJD_TDB'] -= theta_mcmc[0] # center midtransit on time 0

bjd_tdb = []
rel_flux = []
rel_flux_err = []

for i in range(len(dates)):
    bjd_tdb.extend(data_arr[i]['BJD_TDB'].to_numpy())
    rel_flux.extend(data_arr[i]['rel_flux_T1_normalized'].to_numpy())
    rel_flux_err.extend(data_arr[i]['rel_flux_err_T1_normalized'].to_numpy())

bjd_tdb = np.array(bjd_tdb)
rel_flux = np.array(rel_flux)
rel_flux_err = np.array(rel_flux_err)

indices = np.argsort(bjd_tdb)
bjd_tdb = bjd_tdb[indices]
rel_flux = rel_flux[indices]
rel_flux_err = rel_flux_err[indices]