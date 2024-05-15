import numpy as np
import matplotlib.pyplot as plt
from math import radians
import pandas as pd
import numpy as np

# Data
theta_values = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    171, 172, 173, 174, 175, 176, 177, 178, 179,
    180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
    351, 352, 353, 354, 355, 356, 357, 358, 359
]

RCS_values = [
    4.878993, 4.87542, 4.86396, 4.8446471, 4.816653, 4.780558, 3.06068, -0.49486363, -6.6792925, -42.66708,
    -43.133201, -6.732148, -0.583348, 3.045994, 4.779143, 4.8143713, 4.842693, 4.8626057, 4.873962, 4.878993,
    4.87542, 4.86396, 4.8446471, 4.816653, 4.780558, 3.06068, -0.49486363, -6.6792925, -42.66708,
    -43.133201, -6.732148, -0.583348, 3.045994, 4.779143, 4.8143713, 4.842693, 4.8626057, 4.873962
]

# Find the index of the closest value to 4.8 near 0 degrees
theta2 = np.deg2rad(theta_values)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_theta_zero_location('N')
ax.plot(theta2, RCS_values, label="LHFT [dBsm]", color='black', linestyle='dashdot', linewidth=2)

ax.set_theta_direction(-1)
#ax.set_rticks(np.linspace(0, np.max(r), 5))  # Adjust radial ticks based on data
#ax.set_rlabel_position(0)  # Move radial labels away from plotted line
ax.set_rlim(-60, 20)
ax.set_rticks([-60, -50, -40, -30, -20, -10, 0, 10, 20]) 
ax.grid(True)
ax.set_title("Polar Plot for RCS (dBsm) with Theta", va='bottom')
plt.show()








# import numpy as np

# gt = 6
# Antenna_gain_dBsm = 6
# gt_linear = 10**(gt / 10)
# print(gt_linear)
# range_antenna_plate = 0.8

# MoM_RCS_0_deg = 4.98632452    # dBsm

# # Calculate the constant to normalize the peak at 0°


# Pt = (1200*600)*60
# Power_ratio = (162232/Pt)**2
# print(f" power ratio {Power_ratio}")
# RCS_without_const = Power_ratio*(4*np.pi*(range_antenna_plate**2))**2
# print(f" RCS without const factor {RCS_without_const}") 
# constant = RCS_without_const / (10 ** (MoM_RCS_0_deg / 10))
# # Converting gains from dB to linear scale
# Antenna_gain_linear = 10**((Antenna_gain_dBsm) / 10)
# print(f" Antenna_gain_linear  {constant}") 
# RCS_final = (RCS_without_const / constant)
# print(f" RCS with const factor linear {RCS_final}") 
# RCS_dBsm = 10 * np.log10(RCS_final) 

# print(f" RCS with const factor dBsm {RCS_dBsm}")
# pi = np.pi
# print(pi)
















# import numpy as np

# def calculate_coordinates(radius, degree):
#     # Convert degree to radians for calculation
#     angle_rad = np.deg2rad(degree)
    
#     # Calculate x and y coordinates in meters
#     x = np.float32(radius * np.cos(angle_rad))
#     y = np.float32(radius * np.sin(angle_rad))
    
#     return (x, y)

# # Example usage
# radius_input = 0.3  # radius in meters
# degree_input = 180
#    # angle in degrees

# # Get the coordinates
# x, y = calculate_coordinates(radius_input, degree_input)
# print(f"For radius = {radius_input} m and degree = {degree_input}°:")
# print(f"x-coordinate: {x} m, y-coordinate: {y} m")
# import numpy as np

# # Define the radius of the circle in meters
# radius = 0.8  # in meters

# # Define the increment in degrees
# degree_increment = 5

# # Calculate the number of points around the circle
# num_points = int(360 / degree_increment)

# # List to store coordinates
# coordinates = []

# for i in range(num_points):
#     # Convert degrees to radians for calculation
#     angle_rad = np.deg2rad(i * degree_increment)
    
#     # Calculate x and y coordinates in meters, rounding to minimize floating point errors
#     x = round(radius * np.cos(angle_rad), 8)
#     y = round(radius * np.sin(angle_rad), 8)
    
#     # Append the coordinates to the list
#     coordinates.append((x, y, 0))  # z is always 0

# # Print out all coordinates and calculate back the radius to check accuracy
# print("Coordinates and recalculated radii:")
# for coord in coordinates:
#     recalculated_radius = round(np.sqrt(coord[0]**2 + coord[1]**2), 6)
#     print(f"x: {coord[0]:.8f}, y: {coord[1]:.8f}, z: {coord[2]}, recalculated radius: {recalculated_radius}")


