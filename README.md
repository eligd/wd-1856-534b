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
[]