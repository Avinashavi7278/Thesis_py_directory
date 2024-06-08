import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to read data from the provided text file
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

# Path to the text file
azimuth_txt_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/LHFT_RCS_Results_Processing/azimuth_radiation_pattern.txt'
elevation_txt_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/LHFT_RCS_Results_Processing/elevation_radiation_pattern.txt'
Radiation_pattern_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/LHFT_RCS_Results_Processing/Radiation_pattern_new.txt'


# Read the data from the file
az_angles_rx1, az_values_rx1, az_angles_rx2, az_values_rx2, az_angles_rx3, az_values_rx3, az_angles_rx4, az_values_rx4 = read_data(azimuth_txt_file)
el_angles_rx1, el_values_rx1, el_angles_rx2, el_values_rx2, el_angles_rx3, el_values_rx3, el_angles_rx4, el_values_rx4 = read_data(elevation_txt_file)

# Defining a common set of angles for interpolation
az_common_angles = np.linspace(min(az_angles_rx1.min(), az_angles_rx2.min(), az_angles_rx3.min(), az_angles_rx4.min()),
                            max(az_angles_rx1.max(), az_angles_rx2.max(), az_angles_rx3.max(), az_angles_rx4.max()), 120)

el_common_angles = np.linspace(min(el_angles_rx1.min(), el_angles_rx2.min(), el_angles_rx3.min(), el_angles_rx4.min()),
                            max(el_angles_rx1.max(), el_angles_rx2.max(), el_angles_rx3.max(), el_angles_rx4.max()), 60)


# Interpolate the azimuth values to the common set of angles
az_interp_values_rx1 = np.interp(az_common_angles, az_angles_rx1, az_values_rx1)
az_interp_values_rx2 = np.interp(az_common_angles, az_angles_rx2, az_values_rx2)
az_interp_values_rx3 = np.interp(az_common_angles, az_angles_rx3, az_values_rx3)
az_interp_values_rx4 = np.interp(az_common_angles, az_angles_rx4, az_values_rx4)

# Interpolate the elevation values to the common set of angles
el_interp_values_rx1 = np.interp(el_common_angles, el_angles_rx1, el_values_rx1)
el_interp_values_rx2 = np.interp(el_common_angles, el_angles_rx2, el_values_rx2)
el_interp_values_rx3 = np.interp(el_common_angles, el_angles_rx3, el_values_rx3)
el_interp_values_rx4 = np.interp(el_common_angles, el_angles_rx4, el_values_rx4)


# Calculating the average of the interpolated values
az_average_values = np.mean([az_interp_values_rx1, az_interp_values_rx2, az_interp_values_rx3, az_interp_values_rx4], axis=0)

el_average_values = np.mean([el_interp_values_rx1, el_interp_values_rx2, el_interp_values_rx3, el_interp_values_rx4], axis=0)


# Create the 120x60 matrix
matrix = np.zeros((120, 60))

# Insert the azimuth values in the center row
center_row = 60
matrix[center_row, :] = el_average_values

# Insert the elevation values in the center column
center_col = 30
matrix[:, center_col] = az_average_values

# Compute the rest of the matrix by element-wise addition
for i in range(120):
    for j in range(60):
        if i != center_row and j != center_col:
            matrix[i, j] = matrix[center_row, j] + matrix[i, center_col]
            matrix[i, j] /= 2  # Divide each element by 2

# Save the center value before modifying the center row and column
center_value_matrix = matrix[center_row, center_col]

# Add 95.5 to each value in the center row and center column, except the center value
matrix[center_row, :] += 95.5
matrix[:, center_col] += 95.5

# Restore the center value in the matrix
matrix[center_row, center_col] = center_value_matrix

# Divide the center row and center column values by 2, except the center value
matrix[center_row, :] /= 2
matrix[:, center_col] /= 2

# Restore the center value in the matrix again as it got affected in the previous step
matrix[center_row, center_col] = center_value_matrix

# Apply 2x2 window function near the center row and column
def apply_window_function_near_center(matrix, center_row, center_col):
    smoothed_matrix = np.copy(matrix)
    for i in range(center_row - 1, center_row + 1):
        for j in range(center_col - 1, center_col + 1):
            if i + 1 < matrix.shape[0] and j + 1 < matrix.shape[1]:
                window = matrix[i:i+4, j:j+4]
                window_avg = np.mean(window)
                smoothed_matrix[i:i+4, j:j+4] = window_avg
    return smoothed_matrix

matrix = apply_window_function_near_center(matrix, center_row, center_col)




# # Prepare the data for saving
# azimuths = np.linspace(-60, 60, 120)
# elevations = np.linspace(-30, 30, 60)
# data_to_save = []

# for i in range(120):
#     for j in range(60):
#         theta = azimuths[i]
#         phi = elevations[j]
#         value = matrix[i, j]
#         data_to_save.append(f"{theta:.3f}       {phi:.3f}      {value:.3e}")

# # Save the data to a text file
# header = "Theta [deg.]  Phi [deg.]  Abs(Grlz)[dBi   ]\n" + "-"*46
# with open(Radiation_pattern_file, "w") as file:
#     file.write(header + "\n")
#     for line in data_to_save:
#         file.write(line + "\n")





# Print the highest values and their indices
max_az_value = np.max(matrix[:, center_col])
max_az_index = np.argmax(matrix[:, center_col])

max_el_value = np.max(matrix[center_row, :])
max_el_index = np.argmax(matrix[center_row, :])

print(f"Highest value in new_az_values: {max_az_value} at index {max_az_index}")
print(f"Highest value in new_el_values: {max_el_value} at index {max_el_index}")

# Print the center values of both arrays and the final matrix
center_value_az = az_average_values[center_row]
center_value_el = el_average_values[center_col]

print(f"Center value in az_average_values: {center_value_az}")
print(f"Center value in el_average_values: {center_value_el}")
print(f"Center value in the final matrix: {center_value_matrix}")

# Prepare the grid for plotting
X, Y = np.meshgrid(np.linspace(-60, 60, 120), np.linspace(-30, 30, 60))
X, Y = X.T, Y.T  # Transpose to match the matrix layout

# Plot the 3D surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, matrix, cmap='viridis')

ax.set_xlabel('Azimuth (degrees)')
ax.set_ylabel('Elevation (degrees)')
ax.set_zlabel('Radiation Pattern')

plt.show()





# # Plot the averaged data
# plt.plot(az_common_angles, az_average_values, label='Azimuth Average Pattern')
# plt.xlabel('Angle [degrees]')
# plt.ylabel('dBFS')
# plt.title('Averaged Azimuth Antenna Gain Pattern')
# plt.legend()
# plt.grid(True)
# plt.show()


# # Plot the averaged data
# plt.plot(el_common_angles, el_average_values, label='elevation Average Pattern')
# plt.xlabel('Angle [degrees]')
# plt.ylabel('dBFS')
# plt.title('Averaged elevation Antenna Gain Pattern')
# plt.legend()
# plt.grid(True)
# plt.show()