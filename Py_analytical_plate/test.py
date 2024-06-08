import numpy as np
from scipy.integrate import dblquad

# Define the limits for theta and phi
theta = 0.01
phi = 0.01

# Define the integrand
def integrand(theta, phi):
    return np.cos(theta)

# Perform the double integration
A_eff, _ = dblquad(integrand, -phi/2, phi/2, lambda phi: -theta/2, lambda phi: theta/2)

# Calculate the gain factor
G = (4 * np.pi) / A_eff

print(f"Effective Area (A_eff): {A_eff}")
print(f"Gain Factor (G): {G}")
