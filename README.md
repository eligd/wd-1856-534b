# WD 1856+534 b

## Overview
This repository contains the code to accompany our study of exoplanet WD 1856+534 b. We have reduced and fit lightcurves to 16 transit observations obtained with the 1m Nickel Telescope at Lick Observatory. We used the resulting midtransit times to follow up on preliminary evidence of orbital decay in this system, but ultimately found that orbital decay is not needed to explain the observations. Lastly, we have studied data from [INSERT DISCUSSION OF OTHER DATA SOURCES].

## Transits
We observed 16 transits of WD 1856+534b with the 1m Nickel Telescope at Lick Observatory on the following dates. All transits were observed in the R band with 45 second exposures except the 2022-04-01 transit, which was taken in the V band with 45 second exposures, and the 2022-06-02 transit, which was taken in the R band with 120 second exposures. 

- April 1, 2022
- June 2, 2022
- July 31, 2023
- August 14, 2023
- May 13, 2024
- June 6, 2024
- June 13, 2024
- June 23, 2024
- June 30, 2024
- July 7, 2024
- July 14, 2024
- July 24, 2024
- August 7, 2024
- August 14, 2024
- May 17, 2025
- May 24, 2025

## Data Reduction Procedure
- Download flats, biases, and science data from Mount Hamilton data repository (https://mthamilton.ucolick.org/data/)
- Correct FITS file headers with fits_head.py to ensure files are readable by AstroImageJ (AIJ)
- Import science image sequence to AIJ and flip through to make sure there are no obvious problems. Go to Process $\to$ Data reduction facility. In DP Coordinate Converter, update the object ID and observatory ID, and check that the RA and DEC coordinates look right. Then make the CCD Data Processor window look like this and click start:
![CCD Data Processor](imgs/ccd_data_processor.png)
<img src="imgs/ccd_data_processor.png" alt="CCD Data Processor" width="600" class="center"/>
- Click target star (or one of similar size) and go to Analyze $\to$ Plot Seeing Profile. Make sure you’re using the right aperture (three rings). If everything looks good, click “Save Aperture.” If needed, you can adjust the aperture size in multi-aperture measurements.
- Go to Analyze -> Multi-Aperture. Adjust aperture size if desired; under good conditions, (10, 18, 28) often works well. Click “Aperture Settings” to adjust CCD gain, CCD readout noise, and CCD dark current (see https://mthamilton.ucolick.org/techdocs/detectors/dewar2/dewar2_frame.html). Note that these three values depend on read speed, and be sure to convert the dark current to e-/pix/sec if reported in e-/pix/hr. Click Place Apertures, position apertures over target and comparison stars (see below), and click enter.
<img src="imgs/comp_stars.png" alt="Comparison Stars" width="600" class="center"/>

- Check multi-plot reference star settings to make sure all comparison stars are good. If not, replace yellow/red comparison stars and rerun.
- Multi-plot Y-data: select which quantities to plot (always use BJD_TDB on x-axis), add lightcurve under “Fit Mode” for target (bottom option in menu), set legend labels, check "Show Error”
![AIJ Multi-Plot Part 1](imgs/multiplot1.png)
![AIJ Multi-Plot Part 2](imgs/multiplot2.png)
- Multi-plot Main: set title and subtitle, select auto X/Y-range under X/Y-Axis Scaling, enter predicted ingress/egress under V. Marker 1/2 and check the boxes to display, click Copy under Fit and Normalize Region Selection to send the predicted ingress and egress to the lightcurve model
<img src="imgs/multiplot_main.png" alt="AIJ Multi-Plot Main" width="600" class="center"/>

- Data Set 1 Fit Settings: enter orbital period and eccentricity (uncheck Circle), enter one host star parameter, check the first boxes in the rows labeled “Linear LD u1” and “Quad LD u2”, check “Auto Update Fit”, and verify that transit duration (t14 hms) is comparable to the expected value
<img src="imgs/fit_settings.png" alt="Light Curve Fit Settings" width="600" class="center"/>

(Note: AIJ light curve fit is not critical since we'll redo the fit with MCMC.)
- Click on Multi-plot Main and save using “save all (with options).” Save the following values to a spreadsheet, then convert the spreadsheet to .csv format with ASCII (Western) encoding. Make sure no decimals get truncated in the spreadsheet (may need to adjust # decimal points in Excel $\to$ Format $\to$ Cells $\to$ Number).
<img src="imgs/spreadsheet_cols.png" alt="Spreadsheet Columns" width="400" class="center"/>

## Light Curve Fitting (individual transits)
- 