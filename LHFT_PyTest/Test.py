import numpy as np
from scipy.integrate import dblquad
import matplotlib.pyplot as plt


# Single azimuth and elevation angles in radians
azimuth_angle = np.radians(1)  # Example single value, converted to radians
elevation_angle = np.radians(1)  # Example single value, converted to radians

# Perform the double integral over the angular extent
effective_area, error = dblquad(lambda theta, phi:  np.cos(theta),
                                -azimuth_angle / 2, azimuth_angle / 2,  # Azimuth angle range
                                lambda x: -elevation_angle / 2, lambda x: elevation_angle / 2)  # Elevation angle range

# Check if the effective area is not zero to avoid division by zero
if effective_area == 0:
    raise ValueError("Effective area calculated to be zero, which will cause a division by zero error in gain calculation.")

print(f"Effective Area: {effective_area:.4f} square meters")

# Calculate the gain factor
gain_factor = (4 * np.pi) / effective_area

print(f"Gain Factor: {gain_factor:.4f}")

# Plotting for visualization (though with a single point it might be less meaningful)
phi = np.linspace(-azimuth_angle / 2, azimuth_angle / 2, 25)
theta = np.linspace(-elevation_angle / 2, elevation_angle / 2, 25)
phi_grid, theta_grid = np.meshgrid(phi, theta)
rcs_dbsm_grid = np.full_like(phi_grid, 10)  # Create a grid of the RCS value in dBsm for visualization

# plt.figure(figsize=(12, 6))
# plt.pcolormesh(np.degrees(phi_grid), np.degrees(theta_grid), rcs_dbsm_grid, shading='auto', cmap='viridis')
# plt.colorbar(label='RCS (dBsm)')
# plt.xlabel('Azimuth Angle (degrees)')
# plt.ylabel('Elevation Angle (degrees)')
# plt.title('RCS vs Azimuth and Elevation Angles')
# plt.show()
