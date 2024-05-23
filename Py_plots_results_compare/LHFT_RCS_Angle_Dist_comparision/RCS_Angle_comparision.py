import json
import matplotlib.pyplot as plt
import numpy as np
json_file=0

def load_data(json_file):
    # Load data from a JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)
        return data['Measurements']

def extract_common_params(measurements):
    # Extract common parameters from the measurements
    if not measurements:
        return None

    common_params = {
        "Object_range": measurements[0].get("Object_range", "N/A"),
        "Oversamp_factor": measurements[0].get("Oversamp_factor", "N/A"),
        "azimuth_angle": measurements[0].get("azimuth_angle", "N/A"),
        "rx_antenna_rad": measurements[0].get("rx_antenna_rad", "N/A")
    }
    return common_params

def plot_corner_angle(measurements):
    # Extract mesh_angle_up and RCS_without_const_dBsm from the measurements
    theta = [entry['mesh_angle_up'] for entry in measurements]
    rcs_values = [entry['RCS_without_const_dBsm'] for entry in measurements]

    # Convert theta to radians
    theta_rad = np.deg2rad(theta)
    
    # Create a polar plot
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111, projection='polar')
    ax.plot(theta_rad, rcs_values, color='blue', linestyle='-', marker='o', markersize=4, linewidth=0)
    ax.set_title('Corner RCS vs Angle')
    ax.set_ylim(-60, 10)
    ax.grid(True)
    
    common_params = extract_common_params(measurements)
    if common_params:
        specs_text = (
            "Specifications:\n"
            f"- Oversamp_factor: {common_params['Oversamp_factor']}\n"
            f"- rx_antenna_rad: {common_params['rx_antenna_rad']}\n"
            f"- azimuth_angle: {common_params['azimuth_angle']}"
        )
        plt.figtext(0.73, 0.9, specs_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.4))
    
    plt.show()

def plot_sphere_angle(measurements):
    # Extract mesh_angle_up and RCS_without_const_dBsm from the measurements
    theta = [entry['mesh_angle_r'] for entry in measurements]
    rcs_values = [entry['RCS_without_const_dBsm'] for entry in measurements]

    # Convert theta to radians
    theta_rad = np.deg2rad(theta)
    # theta_rad = [angle * (3.141592653589793 / 180.0) for angle in theta]

    # Create a polar plot
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111, projection='polar')
    ax.scatter(theta_rad, rcs_values, color='blue', s=5)
    ax.set_title('sphere RCS vs Angle ')
    ax.set_ylim(-60, 0)
    ax.grid(True)
    common_params = extract_common_params(measurements)
    if common_params:
        specs_text = (
            "Specifications:\n"
            f"- Oversamp_factor: {common_params['Oversamp_factor']}\n"
            f"- rx_antenna_rad: {common_params['rx_antenna_rad']}\n"
            f"- azimuth_angle: {common_params['azimuth_angle']}"
        )
        plt.figtext(0.73, 0.9, specs_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.4))
    plt.show()

def plot_plate_angle(measurements):
    # Extract Obj_range and RCS_without_const_dBsm from the measurements
    theta = [entry['mesh_angle_r'] for entry in measurements]
    rcs_values = [entry['RCS_without_const_dBsm'] for entry in measurements]

    # Convert theta to radians
    theta_rad = np.deg2rad(theta)
    # theta_rad = [angle * (3.141592653589793 / 180.0) for angle in theta]

    # Create a polar plot
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111, projection='polar')
    ax.scatter(theta_rad, rcs_values, color='blue', s=5)
    ax.set_title('plate RCS vs Angle ')
    ax.set_ylim(-60, 60)
    ax.grid(True)
    common_params = extract_common_params(measurements)
    if common_params:
        specs_text = (
            "Specifications:\n"
            f"- Oversamp_factor: {common_params['Oversamp_factor']}\n"
            f"- rx_antenna_rad: {common_params['rx_antenna_rad']}\n"
            f"- azimuth_angle: {common_params['azimuth_angle']}"
        )
        plt.figtext(0.73, 0.9, specs_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.4))
    plt.show()
# def plot_plate_angle(measurements):
#     # Extract mesh_angle_r and RCS_without_const_dBsm from the measurements
#     theta = [entry['mesh_angle_r'] for entry in measurements]
#     rcs_values = [entry['RCS_without_const_dBsm'] for entry in measurements]

#     # Sort the measurements by theta
#     sorted_measurements = sorted(zip(theta, rcs_values))
#     sorted_theta, sorted_rcs_values = zip(*sorted_measurements)

#     # Convert theta to radians
#     theta_rad = np.deg2rad(sorted_theta)

#     # Create a polar plot
#     plt.figure(figsize=(12, 6))
#     ax = plt.subplot(111, projection='polar')
#     ax.scatter(theta_rad, rcs_values, color='blue', s=10)
#     ax.plot(theta_rad, rcs_values, color='blue')  # Connect the dots with lines
#     ax.set_title('Plate RCS vs Angle')
#     ax.set_ylim(-60, 60)
#     ax.grid(True)

#     # Add specifications text box
#     specs_text = (
#         "Specifications:\n"
#         "- Object_range: 1\n"
#         "- Oversamp_factor: 30\n"
#         "- rx_antenna_radius: 0.3\n"
#         "- azimuth_angle: 5.72"
#     )
#     plt.figtext(0.8, 0.8, specs_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

#     # Display the plot
#     plt.show()


def main():

    Object = "sphere".lower() #sphere, plate, corner

    if Object == "sphere":
        json_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/LHFT_RCS_Angle_Dist_comparision/RCS_sphere_angle_results_auto.json'
        measurements = load_data(json_file)
        plot_sphere_angle(measurements)

    if Object == "corner":
        json_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/LHFT_RCS_Angle_Dist_comparision/RCS_corner_angle_metal_results_auto.json'
        measurements = load_data(json_file)
        plot_corner_angle(measurements)

    if Object == "plate":
        json_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/LHFT_RCS_Angle_Dist_comparision/RCS_plate_angle_results_auto.json'
        measurements = load_data(json_file)
        plot_plate_angle(measurements)

if __name__ == "__main__":
    main()
