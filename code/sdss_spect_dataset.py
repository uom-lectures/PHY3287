from astroML.datasets import fetch_sdss_specgals
import numpy as np
from astropy.io import fits

# get data and save
# fetch_sdss_specgals()


hdulist = fits.open('/Users/andrea/astroML_data/SDSSspecgalsDR8.fit')
data = np.asarray(hdulist[1].data)
np.savez('sdss_spec',data=data)