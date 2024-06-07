import json
import matplotlib.pyplot as plt

def load_data(json_filename, Obj_range, Oversamp_factor):
    # Load data from a JSON file and filter based on Oversamp_factor
    with open(json_filename, 'r') as file:
        data = json.load(file)
        measurements = [entry for entry in data['Measurements'] if entry['Oversamp_factor'] == Oversamp_factor and entry['Obj_range'] == Obj_range]
        return measurements

def extract_common_params(measurements):
    # Extract common parameters from the measurements
    if not measurements:
        return None

    common_params = {
        "Obj_range": measurements[0].get("Obj_range", "N/A"),
        "Oversamp_factor": measurements[0].get("Oversamp_factor", "N/A"),
        "mesh_angle_up": measurements[0].get("mesh_angle_up", "N/A")
    }
    return common_params

def plot_plate_dist(measurements):
    # Extract Obj_range and RCS_without_const_dBsm from the measurements
    rx_antenna_rad = [entry['rx_antenna_rad'] for entry in measurements]
    rcs_values = [entry['RCS_with_const_dBsm'] for entry in measurements]

    # Create a scatter plot
    plt.figure(figsize=(10, 5))
    plt.scatter(rx_antenna_rad, rcs_values, color='blue', s=10)
    plt.title('Plate RCS vs Rx antenna Radius')
    plt.xlabel('Rx antenna Radius in m')
    plt.ylabel('RCS (dBsm)')
    plt.ylim(-40, 40)
    plt.grid(True)

    common_params = extract_common_params(measurements)
    if common_params:
        specs_text = (
            "Specifications:\n"
            f"- Oversamp_factor: {common_params['Oversamp_factor']}\n"
            f"- Plate size is 5cm\n"
            f"- Obj_range: {common_params['Obj_range']}"
        )
        plt.figtext(0.73, 0.9, specs_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.4))

    plt.show()

def plot_corner_dist(measurements):
    # Extract Obj_range and RCS_without_const_dBsm from the measurements
    rx_antenna_rad = [entry['rx_antenna_rad'] for entry in measurements]
    rcs_values = [entry['RCS_without_const_dBsm'] for entry in measurements]

    # Create a scatter plot
    plt.figure(figsize=(12, 6))
    plt.scatter(rx_antenna_rad, rcs_values, color='blue', s=13)
    plt.title('Corner RCS vs Rx antenna Radius')
    plt.xlabel('Rx antenna Radius in m')
    plt.ylabel('RCS (dBsm)')
    plt.ylim(-40, 20)
    plt.grid(True)

    common_params = extract_common_params(measurements)
    if common_params:
        specs_text = (
            "Specifications:\n"
            f"- Obj_range: {common_params['Obj_range']}\n"
            f"- Oversamp_factor: {common_params['Oversamp_factor']}\n"
            f"- mesh_angle_up: {common_params['mesh_angle_up']}\n"
            f"- azimuth_angle: {common_params['azimuth_angle']}"
        )
        plt.figtext(0.73, 0.9, specs_text, fontsize=11, bbox=dict(facecolor='white', alpha=0.4))

    plt.show()


def main():
    Object = "plate".lower()  # sphere, plate, corner
    
    if Object == "plate":
        json_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/LHFT_RCS_Results_Processing/RCS_plate_dist_Rxantenna.json'
        Oversamp_factor = 30  # Specify the desired Oversamp_factor value here
        Obj_range = 3
        measurements = load_data(json_file, Obj_range, Oversamp_factor)
        plot_plate_dist(measurements)

    if Object == "corner":
        Obj_range = 3
        Oversamp_factor = 90
        json_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/LHFT_RCS_Results_Processing/RCS_corner_dist_Rxantenna.json'
        measurements = load_data(json_file, Obj_range, Oversamp_factor)
        plot_corner_dist(measurements)

if __name__ == "__main__":
    main()
