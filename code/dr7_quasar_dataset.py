from astroML.datasets import fetch_dr7_quasar
from astroML.datasets import fetch_sdss_sspp
import numpy as np

# get data
quasars = fetch_dr7_quasar()
stars = fetch_sdss_sspp()

np.savez('dr7_quasar',quasars=quasars,stars=stars)