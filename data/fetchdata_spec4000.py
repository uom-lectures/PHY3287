import numpy as np
import matplotlib.pyplot as plt


def reconstruct_spectra(data):
    spectra = data['spectra']
    coeffs = data['coeffs']
    evecs = data['evecs']
    mask = data['mask']
    mu = data['mu']
    norms = data['norms']
    spec_recons = spectra.copy()
    nev = coeffs.shape[1]
    spec_fill = mu + np.dot(coeffs, evecs[:nev])
    spec_fill *= norms[:, np.newaxis]
    spec_recons[mask] = spec_fill[mask]
    return spec_recons


def compute_wavelengths(data):
    return 10 ** (data['coeff0']
                  + data['coeff1'] * np.arange(data['spectra'].shape[1]))


data = np.load('spec4000.npz')
spectra = reconstruct_spectra(data)
wavelengths = compute_wavelengths(data)

print(spectra.shape)
print(wavelengths.shape)

single_spectrum = spectra[0]
plt.plot(wavelengths,single_spectrum)
# plt.show()

from sklearn import preprocessing
spectra = preprocessing.normalize(spectra)
mu = spectra.mean(0)
std = spectra.std(0)
plt.plot(wavelengths, mu, color='black')
plt.fill_between(wavelengths, mu - std, mu + std, color='#CCCCCC')
plt.xlim(wavelengths[0], wavelengths[-1])
plt.ylim(0, 0.06)
plt.xlabel('wavelength (Angstroms)')
plt.ylabel('scaled flux')
plt.title('Mean Spectrum')

np.random.seed(25255)  # this seed is chosen to emphasize correlation
i1, i2 = np.random.randint(1000, size=2)
plt.scatter(spectra[:, i1], spectra[:, i2])
plt.xlabel('wavelength = %.1f' % wavelengths[i1])
plt.ylabel('wavelength = %.1f' % wavelengths[i2])
plt.title('Random Pair of Spectra Bins')

