      # azimuth_angle_rad = np.deg2rad(azimuth_angle)
      # for i in range(iterations):
      #    Obj_range += 1
      #    From_vector = np.array([Obj_range, 0.0, 0.0])
      #    print(f"The Obj_range is {Obj_range:.2f}")
      #    RCS_without_const_dBsm = simulate_plate(image_width, image_height, Oversamp_factor, Wavelength, 
      #                 rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle,
      #                 mesh_angle_r, Obj_range, render_mode_type)
      #    time.sleep(15)



import numpy as np
import math
# Pi_val = (math.pi)
# print(Pi_val)
# gain_RCS = (4*np.pi)/(3.58**2)
# print(gain_RCS)
azimuth_angle_rad = np.deg2rad(2.86)
print(azimuth_angle_rad)
# Object = "plAte".lower()
# print(Object)
# import numpy as np

# def calculate_rcs(Pt, Pr, r, theta, k):
#     """
#     Calculate the Radar Cross Section based on given parameters.

#     Parameters:
#     - Pt: float, transmitted power
#     - Pr: float, received power
#     - r: float, range (distance to the target)
#     - theta: float, azimuth angle in degrees
#     - k: float, scaling factor for RCS_const

#     Returns:
#     - RCS: float, calculated Radar Cross Section
#     """
#     # Convert theta from degrees to radians for computation
#     theta_rad = np.deg2rad(theta)

#     # Calculate RCS_const inversely proportional to theta
#     RCS_const = k / theta_rad if theta_rad != 0 else float('inf')
#     print(RCS_const)

#     # Calculate RCS using the formula
#     RCS = (Pr / Pt) * ((4 * np.pi * r**2)**2) * RCS_const

#     return RCS

# # Example usage
# Pt = 1.0  # example transmitted power
# Pr = 0.01  # example received power
# r = 100.0  # example range
# theta = 10*2  # example azimuth angle
# k = 1000  # example scaling factor

# rcs_value = calculate_rcs(Pt, Pr, r, theta, k)
# print(f"RCS value: {rcs_value}")
