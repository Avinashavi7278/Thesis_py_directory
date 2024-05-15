import json
import matplotlib.pyplot as plt

def load_data(json_filename):
    # Load data from a JSON file
    with open(json_filename, 'r') as file:
        data = json.load(file)
        return data['Measurements']

def plot_polar_data(measurements):
    # Extract mesh_angle_up and RCS_without_const_dBsm from the measurements
    theta = [entry['mesh_angle_up'] for entry in measurements]
    rcs_values = [entry['RCS_without_const_dBsm'] for entry in measurements]

    # Convert theta to radians
    theta_rad = [angle * (3.141592653589793 / 180.0) for angle in theta]

    # Create a polar plot
    plt.figure(figsize=(10, 5))
    ax = plt.subplot(111, projection='polar')
    ax.scatter(theta_rad, rcs_values, color='blue')
    ax.set_title('RCS vs Mesh Angle Up')
    ax.set_xlabel('Theta (radians)')
    ax.set_ylabel('RCS (dBsm)')
    ax.set_ylim(-60, 0)
    ax.grid(True)
    plt.show()

def main():
    json_filename = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare\RCS_Dist_comparision/RCS_corner_angle_results_auto.json'
    measurements = load_data(json_filename)
    plot_polar_data(measurements)

if __name__ == "__main__":
    main()
