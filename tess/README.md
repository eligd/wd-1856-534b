# TESS Data

## Code
- `tess_lightcurve_exp_20.ipynb` and `tess_lightcurve_exp_120.ipynb`: examples of TESS data (20 and 120 second exposure times) with expected transit times overplotted
    - Data are very noisy, which makes sense given that the WD is much fainter than the targets TESS is designed for
    - Nonetheless, transits are evident in at least the 120 second exposures (see below)
- `phase_folded_lightcurve.ipynb`: phase-folded TESS light curves using all TESS data available with 120 second exposures
<div style="text-align: center;">
    <img src="../imgs/tess_exp_120s.png" alt="TESS light curve with 120s exposures" width="600"/>
</div>

## Background
- TESS Input Catalog (TIC) ID: 267574918.01
- TESS Object of Interest (TOI): 1690.01
- 74 observations available on MAST: 
    - 23 Full Frame Images (FFI), 30 minute cadence
        - 5 with 24 minute exposure time, 6 with 8 minute exposure time, 12 with 2.6 minute exposure time
    - 51 TOI observations
        - 33 with 2 minute exposure time, 18 with 20 second exposure time