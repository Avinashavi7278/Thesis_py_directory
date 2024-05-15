import numpy as np

def calculate_azimuth_angle(width, distance):
    """
    Calculate the azimuth angle to ensure the rays from the Tx antenna exactly cover the plate width.

    Parameters:
    - width: float, the width of the plate in meters
    - distance: float, the distance from the Tx antenna to the plate in meters

    Returns:
    - theta_degrees: float, the azimuth angle in degrees
    """
    # Calculate the half-angle theta/2 in radians
    theta_half_radians = np.arctan((width / 2) / distance)
    
    # Calculate the full theta in radians
    theta_radians = 2 * theta_half_radians
    
    # Convert theta from radians to degrees
    theta_degrees = np.degrees(theta_radians)
    
    return theta_degrees

# Constants
plate_width = 0.1  # in meters
antenna_distance = 0.8  # in meters

# Calculate the azimuth angle
theta = calculate_azimuth_angle(plate_width, antenna_distance)
print(f"The azimuth angle should be set to {theta:.2f} degrees.")



# To continue from where your code snippet leaves off and compute the Radar Cross Section (RCS) in Python, we need to focus on translating the radar signal processing
#  results (particularly, the FFT range and angle data) into RCS values. I'll outline how to integrate RCS calculation into your existing pipeline, considering you've
# already done substantial preprocessing, FFT, and normalization of your radar data.

# import matplotlib.pyplot as plt
# import numpy as np
# # Constants and radar specifications taken from IWR6843AOP user guide

# c = 299792458  # Speed of light in meters/second
# transmitted_power = 10  # Transmitted power in dBm(IWR6843AOP max transmit power)
# Gt = 6  # Gain of the transmitting antenna in dBi
# Gr = 6  # Gain of the receiving antenna in dBi
# carrier_frequency = 60e9  # Carrier frequency in Hz (example for automotive radar)
# wavelength = 0.00495
# bandwidth = 4e9

# # Converting gains from dB to linear scale
# Gt_linear = 10**(Gt / 10)
# Gr_linear = 10**(Gr / 10)

# range_resolution = c / (2 * bandwidth)  # Bandwidth of the chirp
# # accessing the first row of this array, to find the length of array 
# range_bins = np.arange(len(fft_range_angle_abs_norm_log[0])) * range_resolution 


# # The maximum value within each row is assumed to be the return from a target, to find the range
# # bin that has the maximum response for each row.
# target_bin = np.argmax(fft_range_angle_abs_norm_log, axis=1)
# target_range = range_bins[target_bin]

# # Calculating received power Pr from the normalized FFT log data
# Pr = 10**((fft_range_angle_abs_norm_log[target_bin] - 30) / 10)  # Convert dBm to watts
# #Converting transmitted_power inot watts
# Pt = 10**((transmitted_power - 30) / 10)

# # RCS calculation
# RCS = (Pr * (4 * np.pi)**3 * target_range**2) / (Pt * Gt_linear * Gr_linear * wavelength**2)
# RCS_dBsm = 10 * np.log10(RCS)  # Convert RCS to dBsm



# # Compute angles assuming uniform spacing across the FFT indices
# angles = np.linspace(-np.pi / 2, np.pi / 2, angular_dim)

# plt.figure(figsize=(10, 5))
# plt.plot(angles, RCS_dBsm)
# plt.xlabel('Angle (radians)')
# plt.ylabel('RCS (dBsm)')
# plt.title('RCS vs. Angle')
# plt.grid(True)
# plt.show()
# import numpy as np

# gt = 6
# Antenna_gain_dBsm = 6
# gt_linear = 10**(gt / 10)
# print(gt_linear)
# range_antenna_plate = 0.8

# Pt = (1200*600)*60
# Power_ratio = (162232/Pt)**2
# print(f" power ratio {Power_ratio}")
# RCS_without_const = Power_ratio*(4*np.pi*(range_antenna_plate**2))**2
# print(f" RCS without const factor {RCS_without_const}") 
# # Converting gains from dB to linear scale
# Antenna_gain_linear = 10**((-6) / 10)
# print(f" Antenna_gain_linear  {Antenna_gain_linear}") 
# RCS_final = (RCS_without_const / Antenna_gain_linear)
# print(f" RCS with const factor linear {RCS_final}") 
# RCS_dBsm = 10 * np.log10(RCS_final) 

# print(f" RCS without const factor dBsm {RCS_dBsm}")
# pi = np.pi
# print(pi)

# import numpy as np

# # Define the radius of the circle
# radius = 80  # in cm

# # Define the increment in degrees
# degree_increment = 5

# # Calculate the number of points around the circle
# num_points = int(360 / degree_increment)

# # List to store coordinates
# coordinates = []

# for i in range(num_points):
#     # Convert degrees to radians for calculation
#     angle_rad = np.deg2rad(i * degree_increment)
    
#     # Calculate x and y coordinates
#     x = radius * np.cos(angle_rad)
#     y = radius * np.sin(angle_rad)
    
#     # Append the coordinates to the list
#     coordinates.append((x, y, 0))  # z is always 0

# # Print out all coordinates
# for coord in coordinates:
#     print(f"x: {coord[0]:.2f}, y: {coord[1]:.2f}, z: {coord[2]}")
