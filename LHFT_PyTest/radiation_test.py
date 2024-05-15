import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming the script_directory is defined or you can use an explicit path
script_directory = os.path.dirname(__file__)
radiation_pattern_filename = os.path.join(script_directory, "../LHFT_PyTest/Ant_Pattern_onChip_310GHz_Einzelelement_v2-unmodifiedCopy.txt")

def load_radiation_pattern_from_cst(filename):
    # Attempt to load the file, skipping non-numeric lines
    try:
        # Initially attempt to load data assuming file is formatted correctly
        data = np.loadtxt(filename, comments=['#', '----------'], dtype=float)
    except ValueError:
        # If ValueError occurs, reattempt to load data with additional skips
        data = np.loadtxt(filename, skiprows=2, comments=['#', '----------'], dtype=float)

    phi = data[:, 0]   # assuming the first column is phi
    theta = data[:, 1] # assuming the second column is theta
    gain = data[:, 2]  # assuming the third column is gain

    # Convert phi, theta from degrees to radians for plotting
    phi_rad = np.radians(phi)
    theta_rad = np.radians(theta)

    # Convert spherical coordinates to Cartesian coordinates for the plot
    x = gain * np.sin(theta_rad) * np.cos(phi_rad)
    y = gain * np.sin(theta_rad) * np.sin(phi_rad)
    z = gain * np.cos(theta_rad)

    return x, y, z, phi, theta

def plot_radiation_pattern_3d(x, y, z, subsampling_factor=1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[::subsampling_factor], y[::subsampling_factor], z[::subsampling_factor], c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title('3D Radiation Pattern')
    plt.show()

# Load the radiation pattern data
x, y, z, phi_axis, theta_axis = load_radiation_pattern_from_cst(radiation_pattern_filename)

# Plot the radiation pattern
plot_radiation_pattern_3d(x, y, z, subsampling_factor=8)
