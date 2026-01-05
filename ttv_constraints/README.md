# TTV constraints

## Code
- `grid_search.py`: perform grid search in 3D phase space of period, mass, and mean anomaly; save $\chi^2$ for each set of parameter values
- `get_transit_times.py`: compute and plot transit times using best hypothetical companion from prior run of `grid_search.py`
- `mcmc_ttv.py`: TTV constraints exploring full 7D phase space with MCMC
    - Yields poor constraints, but still useful for checking if 3D grid search is missing promising regions of 7D phase space
- `utils.py`: utility functions, currently only for implementing Rømer delay correction (see below)
- `utils_plot.py`: plotting functions
- `config_files`: directory for sample configuration files

## Inputs (set in config file)
- Filepath of csv containing observed transit times
- $\text{BJD}_{\text{TDB}}$ of reference transit
- Universal gravitational constant
- Stellar mass (includes planet mass as in Kubiak et al.)
- WD 1856+534 b parameters:
    - Mass
    - Period
    - Eccentricity
    - Inclination
    - Longitude of ascending node
    - Argument of periastron
    - Mean anomaly
- Grid search parameters:
    - Assume $e=0$, which means that argument of periastron can be fixed at 0
    - Assume inclination is same as WD 1856+534 b, which means that longitude of ascending node can be fixed at 0
    - Minimum mass, maximum mass, and number of mass points
    - Minimum period, maximum period, and number of period points
    - Minimum mean anomaly, maximum mean anomaly, and number of mean anomaly points
- Integration time step
- Integration start epoch
- Integration duration

## Instructions
Start by setting parameters in config file. Then run grid search with the command

`python grid_search.py --config_file <config_name>.yml`

where `<config_name>` is replaced by the name of the configuration file. After the grid search is complete, plot the transit timing residuals with

`python get_transit_times.py --config_file <config_name>.yml`

using the *same* config file.

## Current approach; based on Kubiak et al. (2023)
- Include WD 1856+534 b mass in stellar mass (rather than planet mass) following Kubiak et al. (2023)
    - Kubiak et al. (2023) use $10M_J$ planetary mass, whereas I used $5.2M_J$ based on results of Limbach et al. (2025)
- WD 1856+534 b period set to best fit value obtained by fitting constant period model to observed transit times
- WD 1856+534 b inclination set to value from discovery paper Vanderburg et al. (2020)
- WD 1856+534 b eccentricity set to 0
- WD 1856+534 b longitude of ascending node set to 0 (defines reference plane)
- WD 1856+534 b argument of periastron set to 0 (this parameter has no physical meaning when $e=0$, but needs to be set for ttvfast to run)
- WD 1856+534 b mean anomaly initialized to 0
- Start by considering only one additional outer planet
- Outer planet's mass, period, eccentricity, inclination, longitude of ascending node, argument of periastron, and mean anomaly are free parameters
    - Currently setting inclination to WD 1856+534b's inclination and setting eccentricity, argument of periastron, and longitude of ascending node to 0
    - This makes the grid search tractable (only three dimensions)
- Numerical integration time step of 0.02 days, following Kubiak et al. (2023)
- Integrate for 2200 days (enough to cover baseline of observations)
- Chi-squared test statistic computed from transit time uncertainties $\sigma_i$, observed TTVs $y_i$, and modeled TTVs $m_i$ as
$$\chi^2 = \sum_i\frac{(y_i-m_i)^2}{\sigma_i^2}$$
- Likelihood function defined as $\mathcal{L} = e^{-\chi^2/2}$
- Likelihood ratio plot:
    - Collapse grid along mean anomaly axis by choosing value of mean anomaly that leads to highest likelihood for each combination of outer planet mass and period
    - Heatmaps show log base 10 of ratio of likelihood with hypothetical outer planet to likelihood under constant period model

## Rømer delay correction
- Based on Equations 6 and 7 of https://arxiv.org/abs/1302.0563

$$A_{\text{Roem}} = \frac{G^{\frac{1}{3}}}{c(2\pi)^{\frac{2}{3}}} P_{\text{trip}}^{\frac{2}{3}}\left[\frac{M_3\sin{i_{\text{trip}}}}{M_{\text{trip}}^{\frac{2}{3}}}\right]$$
$$\frac{\mathcal{R}(t)}{A_{\text{Roem}}} \simeq \left[(1-e^2)^{\frac{1}{2}}\sin{u(t)}\cos{\omega} + (\cos{u(t)} - e)\sin{\omega}\right]$$

- In our case $P_{\text{trip}} = P_{\text{outer}}$, $M_3 = M_{\text{outer}}$, $M_{\text{trip}} = M_{\text{total}}$, and $i_{\text{trip}} = i_{\text{inner}} = i_{\text{outer}}$
- For now we are assuming $e=0$ and $\omega = 0$, so the second equation simplifies to
$$\frac{\mathcal{R}(t)}{A_{\text{Roem}}} \simeq \sin{u(t)}$$
- For a non-eccentric orbit, the eccentric anomaly $u$ is equal to the mean anomaly $M$, which can be calculated from the reference transit $t_0$ and the period $P_{\text{inner}}$ of WD 1856+534 b as
$$u(t)=M(t)=\frac{2\pi}{P_{\text{inner}}}\cdot (t-t_0)$$