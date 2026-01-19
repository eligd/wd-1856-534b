import numpy as np
import pandas as pd
from astroquery.vizier import Vizier

# read in all_times.csv (data from Nickel and the literature)
df = pd.read_csv('../all_times.csv')
t = df['T_mid'].to_numpy()
epoch = df['Orbit number'].to_numpy()
uncertainty = df['Uncertainty (days)'].to_numpy()

# read in TESS transit times
# see README at https://cdsarc.cds.unistra.fr/ftp/J/A+A/705/A5/ReadMe
vizier = Vizier(row_limit=-1)
catalog = vizier.query_constraints(catalog='J/A+A/705/A5/oc-list', columns=['*'], TOI=str('TOI-1690'))
bjd = np.array(catalog[0]['TimePred'])
o_c = np.array(catalog[0]['O-C'])
tess_upper_err = np.array(catalog[0]['E_O-C'])
tess_lower_err = np.array(catalog[0]['e_O-C'])

# convert BJD to epoch number
# note that difference between BJD and BJD_TDB isn't large enough to affect epoch computed here
def get_epoch(midtransit, period=1.4079392114, ref_transit=2458779.3750830120):
    epoch = np.around((midtransit - ref_transit) / period)
    return epoch
tess_epoch = get_epoch(bjd)

tess_t = o_c + bjd # reconstruct Naponiello transit times (O-C + TimePred)

# combine Nickel/literature data with TESS data
combined_t = np.concatenate([t, tess_t])
combined_epoch = np.concatenate([epoch, tess_epoch])
combined_upper_err = np.concatenate([uncertainty, tess_upper_err])
combined_lower_err = np.concatenate([uncertainty, tess_lower_err])
combined_df = pd.DataFrame({'t': combined_t, 'Epoch': combined_epoch, 'Upper Error': combined_upper_err, 'Lower Error': combined_lower_err})
combined_df.to_csv('../all_times_combined.csv')