from astropy.io import fits
import glob

dates = ['2022-04-01', '2022-06-02', '2023-07-31', '2023-08-14', '2024-05-13', 
         '2024-06-06', '2024-06-13', '2024-06-23', '2024-06-30', '2024-07-07',
         '2024-07-14', '2024-07-24', '2024-08-07', '2024-08-14', '2025-05-17',
         '2025-05-24', '2025-06-07', '2025-06-17', '2025-07-01', '2025-07-18']
base_dir = '/Users/eligendreaudistler/Desktop/wd-1856-534b/data'

for date in dates:
    files = glob.glob('{}/{}/*/*.fits'.format(base_dir, date))
    for file in files:
        data, header = fits.getdata(file, header=True)
        datebeg = header['DATE-BEG']
        header['DATE-OBS'] = datebeg
        fits.writeto(file, data, header, overwrite=True)
    print("Finished {}".format(date))