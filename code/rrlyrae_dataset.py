from astroML.datasets import fetch_rrlyrae_combined
import numpy as np

# get data and save
data, labels = fetch_rrlyrae_combined()
np.savez('rrlyrae',data=data,labels=labels)