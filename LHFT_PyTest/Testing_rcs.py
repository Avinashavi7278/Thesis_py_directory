import numpy as np
import matplotlib.pyplot as plt

# # Define the data for each TX-RX pair (manually included based on the provided data)
# Define the data for each TX-RX pair
tx3_rx1 = np.array([
    [-30.191004442575114, 72.70642201834862],
    [-29.33314055887069, 73.41284403669724],
    [-27.96077915932809, 74.63302752293578],
    [-26.33148445098709, 76.23853211009174],
    [-25.045556110512337, 77.65137614678898],
    [-23.844956757183034, 78.80733944954127],
    [-22.472122183959407, 79.83486238532109],
    [-21.444231224205456, 81.31192660550458],
    [-20.330222654504347, 82.72477064220183],
    [-19.043978864908908, 84.0091743119266],
    [-17.843694960700294, 85.29357798165137],
    [-16.5576088956652, 86.64220183486238],
    [-15.271365106069759, 87.92660550458714],
    [-13.299019478983197, 89.8532110091743],
    [-11.669724770642194, 91.45871559633026],
    [-10.039956888620164, 92.87155963302752],
    [-8.409873557477454, 94.15596330275228],
    [-6.864488315238816, 94.92660550458714],
    [-4.372440261822767, 95.24770642201834],
    [-2.1371677926447674, 95.11926605504587],
    [-0.41781236034803726, 95.05504587155963],
    [1.2149522883205037, 95.24770642201834],
    [3.9645645487763304, 95.69724770642202],
    [6.2845351068583994, 96.08256880733944],
    [7.402802239688768, 95.76146788990825],
    [9.12420809126995, 94.86238532110092],
    [10.156988512394534, 94.348623853211],
    [11.275886543466271, 93.77064220183486],
    [13.94490155358693, 92.03669724770641],
    [15.064430482900043, 91.20183486238531],
    [15.840119870665866, 90.36697247706421],
    [16.444678110459776, 89.21100917431193],
    [18.255829236876018, 86.77064220183486],
    [19.03199179832287, 85.74311926605503],
    [20.668857285560335, 84.26605504587155],
    [21.100549407218537, 83.4954128440367],
    [22.134749349386198, 82.40366972477064],
    [23.428248468757396, 80.73394495412843],
    [24.89256328697985, 79.5137614678899],
    [26.186377855471733, 77.71559633027522],
    [27.13414473857155, 76.81651376146789],
    [28.08112299886966, 76.23853211009174],
    [28.16913330354093, 75.40366972477064],
    [30.06466706974055, 73.60550458715596]
])

print(f'Length of tx3_rx1: {len(tx3_rx1)}')
tx3_rx2 = np.array([
    [-30.016876527956676, 71.80733944954127],
    [-28.816018572257697, 72.79467511620733],
    [-28.730632738361237, 73.09174311926606],
    [-27.70163770668489, 74.11926605504587],
    [-26.5018269761573, 75.59633027522935],
    [-25.559580452669483, 76.94495412844036],
    [-24.444625535606313, 77.97247706422019],
    [-23.330143792224177, 79.19266055045871],
    [-22.1300176125759, 80.54128440366972],
    [-21.27231145343182, 81.31192660550458],
    [-19.900265503009905, 82.66055045871559],
    [-18.355826608133324, 83.81651376146789],
    [-17.326831576456975, 84.8440366972477],
    [-15.953997003233347, 85.87155963302752],
    [-14.667280039956882, 86.96330275229357],
    [-13.380720801240763, 88.11926605504587],
    [-12.524434163139766, 89.46788990825688],
    [-11.152545937278195, 90.88073394495412],
    [-10.124024079282872, 92.10091743119266],
    [-8.83778028968744, 93.38532110091742],
    [-7.208485581346437, 94.99082568807339],
     [-5.402539365421504, 94.66972477064219],
    [-3.855419153019106, 94.73394495412843],
    [-1.5345022475749843, 94.73394495412843],
    [1.0431902421072081, 95.1834862385321],
    [2.847559212428699, 95.50458715596329],
    [4.221655582135071, 96.0183486238532],
    [6.971110118030552, 96.5321100917431],
    [9.810152204200733, 95.56880733944953],
    [8.002786467232731, 96.46788990825688],
    [11.102074077968517, 94.54128440366972],
    [12.737993217843915, 93.44954128440367],
    [14.202308036066356, 92.22935779816513],
    [16.097368628584952, 90.62385321100916],
    [17.907100233958104, 88.76146788990825],
    [19.457374937567373, 87.54128440366972],
    [21.00938461134041, 85.61467889908256],
    [22.47417260324388, 84.20183486238531],
    [23.596067400962127, 82.40366972477064],
    [24.975684130280484, 80.6697247706422],
    [26.096317131516003, 79.38532110091742],
    [27.474987513472314, 78.03669724770641],
    [28.42370074393419, 76.75229357798165],
    [29.88943508319973, 74.95412844036696]
    
   
    
])
print(f'Length of tx3_rx2: {len(tx3_rx2)}')
tx3_rx3 = np.array([
    [-30.023185510370386, 74.37614678899082],
    [-29.25159696117347, 75.21100917431193],
    [-28.305908926918796, 75.09503118905324],
    [-27.877342866906755, 75.66055045871559],
    [-27.277043190242104, 76.23853211009174],
    [-26.075497489550745, 77.0091743119266],
    [-24.187376777687227, 78.22935779816513],
    [-22.128598091532815, 79.96330275229357],
    [-20.413343497805, 81.56880733944953],
    [-19.64317446965116, 82.98165137614679],
    [-18.786256933308792, 84.07339449541284],
    [-17.84290633789858, 84.97247706422017],
    [-16.47070266291632, 86.25688073394494],
    [-15.270576483268044, 87.60550458715596],
    [-14.927295258113606, 87.76949904053794],
    [-14.327383612418174, 88.56880733944953],
    [-13.297599957940111, 89.27522935779815],
    [-11.323992534370806, 90.6880733944954],
    [-10.20998396466969, 92.10091743119266],
    [-8.236849714781417, 93.70642201834862],
    [-7.3793352261141045, 94.49177267807565],
    [-6.949817302384261, 94.66972477064219],
    [-5.231408217449591, 94.99082568807339],
    [-3.59722404773796, 94.60550458715596],
    [-0.7610210036539513, 94.79816513761467],
    [2.245682290160616, 95.56880733944953],
    [4.9087037669882605, 96.27522935779815],
    [7.486080807549762, 96.8532110091743],
    [8.605294287742183, 96.14678899082568],
    [9.896585263268584, 95.37614678899082],
    [11.016745090823065, 94.28440366972477],
    [12.824110827791074, 93.38532110091742],
    [14.804316795030665, 92.04812473529827],
    [14.804973581136153, 91.8440366972477],
    [17.130306774269876, 90.04587155963303],
    [19.11101180305461, 88.56880733944953],
    [20.835098972161624, 86.57798165137613],
    [23.160432165295354, 84.77981651376146],
    [24.88562340632477, 82.3394495412844],
    [25.920927420414827, 80.79816513761467],
    [26.869640650876704, 79.5137614678899],
    [28.247995583712324, 78.29357798165137],
    [29.11153755158908, 76.6880733944954],
    [30.060250782050957, 75.40366972477064]  
])
print(f'Length of tx3_rx3: {len(tx3_rx3)}')


tx3_rx4 = np.array([
    [-29.92492310927683, 69.36697247706422],
    [-29.068163297494806, 70.5229357798165],
    [-27.95336610499198, 71.61467889908256],
    [-27.01102727800349, 72.86238647236907],
    [-26.754501721826443, 73.47706422018348],
    [-26.4985147603901, 74.24770642201834],
    [-25.469046555032726, 75.08256880733944],
    [-25.041928445624453, 76.1743119266055],
    [-24.098262401093553, 76.94495412844036],
    [-23.327620199258693, 78.1651376146789],
    [-22.042007307904626, 79.70642201834862],
    [-21.099129886175437, 80.79816513761467],
    [-19.89979232932888, 82.46788990825688],
    [-18.95833442864277, 84.13761467889907],
    [-17.415157330248935, 85.80733944954127],
    [-16.215188875161004, 87.22018348623853],
    [-15.272153728871473, 88.24770642201834],
    [-14.49985098272969, 88.72843824672032],
    [-14.243158697195124, 89.27522935779815],
    [-12.956599458479005, 90.43119266055045],
    [-11.239767619147756, 91.39449541284404],
    [-9.61031518624641, 92.93577981651376],
    [-7.550590152729946, 94.28440366972477],
    [-4.888041849583338, 95.1834862385321],
    [-2.31066480902183, 95.76146788990825],
    [-0.0774427591283029, 96.46788990825688],
    [2.759075734076397, 96.5321100917431],
    [5.33755684656029, 96.66055045871559],
    [7.487342604032499, 96.3394495412844],
    [9.640125128151205, 94.79816513761467],
    [10.932362451039673, 93.64220183486238],
    [12.567650692673702, 92.80733944954127],
    [14.032438684577173, 91.39449541284404],
    [16.18537893325623, 89.78899082568807],
    [18.424594516442795, 88.05504587155963],
    [20.31870876159934, 86.83486238532109],
    [22.299413790384072, 85.35779816513761],
    [23.1616939617781, 84.26605504587155],
     [24.196051628506098, 83.11009174311926],
    [24.885781130885107, 82.27522935779815],
    [26.351200021029953, 80.60550458715596],
    [27.730659025787972, 78.93577981651376],
    [29.28219552587997, 77.20183486238531],
    [30.0582003627665, 76.23853211009174]   
])
print(f'Length of tx3_rx4: {len(tx3_rx4)}')

# # Combine all data into one array for sorting
# # Round the data to 4 decimal places
# tx3_rx1 = np.round(tx3_rx1, 4)
# tx3_rx2 = np.round(tx3_rx2, 4)
# tx3_rx3 = np.round(tx3_rx3, 4)
# tx3_rx4 = np.round(tx3_rx4, 4)
# combined_data = np.hstack((tx3_rx1, tx3_rx2, tx3_rx3, tx3_rx4))

# # Sort data by the first column (angle)
# sorted_indices = np.argsort(combined_data[:, 0])
# sorted_data = combined_data[sorted_indices]

# # Split sorted data back into separate arrays
# sorted_tx3_rx1 = sorted_data[:, :2]
# sorted_tx3_rx2 = sorted_data[:, 2:4]
# sorted_tx3_rx3 = sorted_data[:, 4:6]
# sorted_tx3_rx4 = sorted_data[:, 6:8]

# # Save sorted data to a text file in the specified format
# txt_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/LHFT_PyTest/elevation_radiation_pattern.txt'
# with open(txt_file, "w") as file:
#     file.write("angle1       TX3RX1     ||   angle2      TX3RX2     ||   angle3      TX3RX3     ||   angle4     TX3RX4\n")
#     for i in range(len(sorted_tx3_rx1)):
#         file.write(f"{sorted_tx3_rx1[i, 0]}     {sorted_tx3_rx1[i, 1]}    ||    {sorted_tx3_rx2[i, 0]}      {sorted_tx3_rx2[i, 1]}    ||    {sorted_tx3_rx3[i, 0]}      {sorted_tx3_rx3[i, 1]}    ||    {sorted_tx3_rx4[i, 0]}      {sorted_tx3_rx4[i, 1]}\n")
#     print("Data save to txt file success ")

def read_data(txt_file):
    with open(txt_file, 'r') as file:
        lines = file.readlines()
    
    angles_rx1, values_rx1 = [], []
    angles_rx2, values_rx2 = [], []
    angles_rx3, values_rx3 = [], []
    angles_rx4, values_rx4 = [], []

    for line in lines[1:]:  # Skip the header
        parts = line.split('||')
        angle_rx1, value_rx1 = map(float, parts[0].split())
        angle_rx2, value_rx2 = map(float, parts[1].split())
        angle_rx3, value_rx3 = map(float, parts[2].split())
        angle_rx4, value_rx4 = map(float, parts[3].split())
        
        angles_rx1.append(angle_rx1)
        values_rx1.append(value_rx1)
        angles_rx2.append(angle_rx2)
        values_rx2.append(value_rx2)
        angles_rx3.append(angle_rx3)
        values_rx3.append(value_rx3)
        angles_rx4.append(angle_rx4)
        values_rx4.append(value_rx4)

    return np.array(angles_rx1), np.array(values_rx1), np.array(angles_rx2), np.array(values_rx2), np.array(angles_rx3), np.array(values_rx3), np.array(angles_rx4), np.array(values_rx4)

# Read the data from the file
txt_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/LHFT_PyTest/elevation_radiation_pattern.txt'

angles_rx1, values_rx1, angles_rx2, values_rx2, angles_rx3, values_rx3, angles_rx4, values_rx4 = read_data(txt_file)

# Define a common set of angles for interpolation
common_angles = np.linspace(min(angles_rx1.min(), angles_rx2.min(), angles_rx3.min(), angles_rx4.min()),
                            max(angles_rx1.max(), angles_rx2.max(), angles_rx3.max(), angles_rx4.max()), 500)

# Interpolate the values to the common set of angles
interp_values_rx1 = np.interp(common_angles, angles_rx1, values_rx1)
interp_values_rx2 = np.interp(common_angles, angles_rx2, values_rx2)
interp_values_rx3 = np.interp(common_angles, angles_rx3, values_rx3)
interp_values_rx4 = np.interp(common_angles, angles_rx4, values_rx4)


# Calculate the average of the interpolated values
average_values = np.mean([interp_values_rx1, interp_values_rx2, interp_values_rx3, interp_values_rx4], axis=0)

# Plot the averaged data
plt.plot(common_angles, average_values, label='Average Pattern')
plt.xlabel('Angle [degrees]')
plt.ylabel('dBFS')
plt.title('Averaged Antenna Gain Pattern')
plt.legend()
plt.grid(True)
plt.show()



# import numpy as np
# from scipy.integrate import dblquad
# import matplotlib.pyplot as plt

# # Example RCS data
# azimuth_angles = np.rad2deg(2.5)  # Azimuth angles in radians
# elevation_angles = np.rad2deg(2.5)  # Elevation angles in radians

# # Create a meshgrid for azimuth and elevation
# phi, theta = np.meshgrid(azimuth_angles, elevation_angles)

# # Example RCS values (replace with actual data)
# # Assuming RCS values are given in dBsm, we need to convert them to linear scale
# rcs_dbsm = np.random.uniform(-40, 30, phi.shape)  # Random example values
# rcs_linear = 10 ** (rcs_dbsm / 10)  # Convert dBsm to linear scale

# # Define the integrand function
# def integrand(phi, theta):
#     # Interpolate RCS values at the given phi, theta
#     rcs_value = np.interp(phi, azimuth_angles, np.mean(rcs_linear, axis=0))
#     return rcs_value * np.cos(theta)

# # Perform the double integral over the angular extent
# effective_area, error = dblquad(integrand, 
#                                 0, np.pi,  # Azimuth angle range
#                                 lambda x: 0, lambda x: np.pi/2)  # Elevation angle range

# print(f"Effective Area: {effective_area:.4f} square meters")

# # Calculate the gain factor
# gain_factor = (4 * np.pi) / effective_area

# print(f"Gain Factor: {gain_factor:.4f}")



# import numpy as np
# from scipy.integrate import dblquad

# # Define the azimuth and elevation angles in radians
# azimuth_angle = np.radians(2.86)  # Example value, convert degrees to radians
# elevation_angle = np.radians(2.86)  # Example value, convert degrees to radians

# # Define the integrand function
# def integrand(phi, theta):
#     return np.cos(phi)

# # Perform the double integral
# effective_area, error = dblquad(integrand, 
#                                 -elevation_angle/2, elevation_angle/2, 
#                                 lambda x: -azimuth_angle/2, lambda x: azimuth_angle/2)

# print(f"Effective Area: {effective_area:.4f} square meters")

# # Calculate the gain factor
# gain_factor = (4 * np.pi) / effective_area

# print(f"Gain Factor: {gain_factor:.4f}")




# import math

# # Given values
# RCS_linear = 4.701194832873645e-05
# desired_RCS_dBsm = -20.5327

# # Calculate the constant k
# k = 10**(desired_RCS_dBsm / 10) / RCS_linear

# # Calculate the new RCS in linear
# RCS_new_linear = k * RCS_linear

# # Convert the new RCS in linear to dBsm
# RCS_new_dBsm = 10 * math.log10(RCS_new_linear)

# # Print the results
# print(f"Constant k: {k}")
# print(f"New RCS in linear: {RCS_new_linear}")
# print(f"New RCS in dBsm: {RCS_new_dBsm}")








# import numpy as np

# def calculate_azimuth_angle(width, distance):
#     """
#     Calculate the azimuth angle to ensure the rays from the Tx antenna exactly cover the plate width.

#     Parameters:
#     - width: float, the width of the plate in meters
#     - distance: float, the distance from the Tx antenna to the plate in meters

#     Returns:
#     - theta_degrees: float, the azimuth angle in degrees
#     """
#     # Calculate the half-angle theta/2 in radians
#     theta_half_radians = np.arctan((width / 2) / distance)
    
#     # Calculate the full theta in radians
#     theta_radians = 2 * theta_half_radians
    
#     # Convert theta from radians to degrees
#     theta_degrees = np.degrees(theta_radians)
    
#     return theta_degrees

# # Constants
# plate_width = 0.1  # in meters
# antenna_distance = 0.8  # in meters

# # Calculate the azimuth angle
# theta = calculate_azimuth_angle(plate_width, antenna_distance)
# print(f"The azimuth angle should be set to {theta:.2f} degrees.")



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
