import numpy as np
import matplotlib.pyplot as plt

def rcs_rect_plate(a, b, freq):
    # Constants
    eps = 0.000001
    lambda_ = 0.0325  # Wavelength in meters
    ka = 2 * np.pi * a / lambda_

    # Compute aspect angle vector
    theta_deg = np.arange(0.05, 85.05, 0.1)
    theta = np.deg2rad(theta_deg)

    # Compute sigma terms for vertical polarization
    sigma1v = np.cos(ka * np.sin(theta)) - 1j * np.sin(ka * np.sin(theta)) / np.sin(theta)
    sigma2v = np.exp(1j * ka - np.pi/4) / (np.sqrt(2 * np.pi) * ka**1.5)
    sigma3v = (1 + np.sin(theta)) * np.exp(-1j * ka * np.sin(theta)) / (1 - np.sin(theta))**2
    sigma4v = (1 - np.sin(theta)) * np.exp(1j * ka * np.sin(theta)) / (1 + np.sin(theta))**2
    sigma5v = 1 - (np.exp(1j * 2 * ka - np.pi/2) / (8 * np.pi * ka**3))

    # Compute sigma terms for horizontal polarization
    sigma1h = np.cos(ka * np.sin(theta)) + 1j * np.sin(ka * np.sin(theta)) / np.sin(theta)
    sigma2h = 4 * np.exp(1j * ka * (np.pi/4)) / (np.sqrt(2 * np.pi * ka))
    sigma3h = np.exp(-1j * ka * np.sin(theta)) / (1 - np.sin(theta))
    sigma4h = np.exp(1j * ka * np.sin(theta)) / (1 + np.sin(theta))
    sigma5h = 1 - (np.exp(1j * 2 * ka + np.pi/4) / (2 * np.pi * ka))

    # Compute vertical polarization RCS
    rcs_v = (b**2 / np.pi) * (np.abs(sigma1v - sigma2v * ((1 / np.cos(theta)) + 0.25 * sigma2v * (sigma3v + sigma4v)) * sigma5v**-1)**2) + eps

    # Compute horizontal polarization RCS
    rcs_h = (b**2 / np.pi) * (np.abs(sigma1h - sigma2h * ((1 / np.cos(theta)) - 0.25 * sigma2h * (sigma3h + sigma4h)) * sigma5h**-1)**2) + eps

    # Compute RCS in dB
    rcsdb_v = 10 * np.log10(rcs_v)
    rcsdb_h = 10 * np.log10(rcs_h)

    # Plotting for vertical polarization
    plt.figure(2)
    plt.plot(theta_deg, rcsdb_v, 'k', label='Eq.(11.50)')
    plt.xticks(np.arange(10, 90, 10))
    plt.title('Vertical Polarization, Frequency = {:.3f} GHz, a = {} m, b = {} m'.format(freq * 1e-9, a, b))
    plt.ylabel('RCS -dBsm')
    plt.xlabel('Aspect angle - deg')
    plt.legend()

    # Plotting for horizontal polarization
    plt.figure(3)
    plt.plot(theta_deg, rcsdb_h, 'k', label='Eq.(11.51)')
    plt.xticks(np.arange(10, 90, 10))
    plt.title('Horizontal Polarization, Frequency = {:.3f} GHz, a = {} m, b = {} m'.format(freq * 1e-9, a, b))
    plt.ylabel('RCS -dBsm')
    plt.xlabel('Aspect angle - deg')
    plt.legend()

    plt.show()

    return rcsdb_v, rcsdb_h

# Example usage
a = 0.1016  # Plate dimension in meters (converted from 10.16cm)
b = 0.1016  # Plate dimension in meters (converted from 10.16cm)
freq = 9e9  # Frequency in Hz (converted from GHz)
rcsdb_v, rcsdb_h = rcs_rect_plate(a, b, freq)