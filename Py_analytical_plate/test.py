# Initialize parameters
iterations = 43
initial_azimuth_angle = 1.43
initial_elevation_angle = 1.43

# Run the process twice
for _ in range(2):
    # Iterate over azimuth angles with increment of 2
    for i in range(iterations):  # azimuth increments by 2
        azimuth_angle = initial_azimuth_angle + i * 2
        # For each azimuth angle, run elevation 20 times with increment of 2
        for j in range(iterations):  # elevation increments by 2
            elevation_angle = initial_elevation_angle + j * 2
            print(f"The azimuth_angle is {azimuth_angle:.2f}")
            print(f"The elevation_angle is {elevation_angle:.2f}")














# # Example list of ray_channel_info objects
# # Replace with your actual list of ray_channel_info objects and ensure mesh_ids is properly defined.
# class RayChannelInfo:
#     def __init__(self, mesh_ids):
#         self.mesh_ids = mesh_ids

# # Replace with actual data
# ray_channel_info_list = [
#     RayChannelInfo([0, 0, 1]),
#     RayChannelInfo([1, 0, 0]),
#     RayChannelInfo([1, 1, 1]),
#     RayChannelInfo([0, 1, 1]),
#     RayChannelInfo([1, -1, -1]),
#     RayChannelInfo([0, 0, 1]),
#     RayChannelInfo([0, 1, 1])
# ]

# filtered_vectors = [
#     (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 1, -1), (1, 0, 0),
#     (1, 0, 1), (1, 0, -1), (1, 1, 0), (1, 1, 1), (1, 1, -1),
#     (1, -1, -1)
# ]

# # Initialize the counter dictionary
# vector_counts = {vec: 0 for vec in filtered_vectors}

# # Iterate over ray_channel_info objects
# for ray_channel_info in ray_channel_info_list:
#     if isinstance(ray_channel_info.mesh_ids, list):
#         tuple_mesh_ids = tuple(ray_channel_info.mesh_ids)  # Convert list to tuple for comparison
#         if tuple_mesh_ids in filtered_vectors:
#             vector_counts[tuple_mesh_ids] += 1

# # If you need the total count of matching rays (x)
# x = sum(vector_counts.values())
# print(f"Total number of matching rays: {x}")






























# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # read the data from the radiation pattern txt file
# def read_data_from_file(file_name):
#     with open(file_name, 'r') as file:
#         lines = file.readlines()
    
#     lines = lines[3:]
    
#     theta = []
#     phi = []
#     abs_grlz = []
    
#     for line in lines:
#         parts = line.split()
#         theta.append(float(parts[0]))
#         phi.append(float(parts[1]))
#         abs_grlz.append(float(parts[2]))
    
#     return np.array(theta), np.array(phi), np.array(abs_grlz)

# # File name
# file_name = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/LHFT_RCS_Results_Processing/Radiation_pattern_new.txt'


# # Read data from the file
# theta, phi, abs_grlz = read_data_from_file(file_name)

# # Create a 3D plot
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# # Plot the data
# sc = ax.scatter(theta, phi, abs_grlz, c=abs_grlz, cmap='viridis')

# # Add labels and title
# ax.set_xlabel('Theta [deg.]')
# ax.set_ylabel('Phi [deg.]')
# ax.set_zlabel('Abs(Grlz) [dBi]')
# ax.set_title('3D Radiation Pattern')

# # Add a color bar
# cbar = plt.colorbar(sc)
# cbar.set_label('Abs(Grlz) [dBi]')

# # Show the plot
# plt.show()






# import numpy as np
# from scipy.integrate import dblquad

# # Define the limits for theta and phi
# theta = 0.01
# phi = 0.01

# # Define the integrand
# def integrand(theta, phi):
#     return np.cos(theta)

# # Perform the double integration
# A_eff, _ = dblquad(integrand, -phi/2, phi/2, lambda phi: -theta/2, lambda phi: theta/2)

# # Calculate the gain factor
# G = (4 * np.pi) / A_eff

# print(f"Effective Area (A_eff): {A_eff}")
# print(f"Gain Factor (G): {G}")

