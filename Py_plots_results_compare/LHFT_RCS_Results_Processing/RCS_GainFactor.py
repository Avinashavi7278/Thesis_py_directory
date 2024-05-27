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
        "mesh_angle_up": measurements[0].get("mesh_angle_up", "N/A"),
        "azimuth_angle": measurements[0].get("azimuth_angle", "N/A"),
        "rx_antenna_rad": measurements[0].get("rx_antenna_rad", "N/A")
    }
    return common_params

def plot_plate_dist(measurements):
    # Extract Obj_range and RCS_without_const_dBsm from the measurements
    rx_antenna_rad = [entry['azimuth_angle'] for entry in measurements]
    rcs_values = [entry['RCS_without_const_dBsm'] for entry in measurements]

    # Create a scatter plot
    plt.figure(figsize=(10, 5))
    plt.scatter(rx_antenna_rad, rcs_values, color='blue', s=13)
    plt.title('Plate RCS vs Azimuth')
    plt.xlabel('Azimuth in degree')
    plt.ylabel('RCS (dBsm)')
    plt.ylim(-40, 40)
    plt.grid(True)

    common_params = extract_common_params(measurements)
    if common_params:
        specs_text = (
            "Specifications:\n"
            f"- Oversamp_factor: {common_params['Oversamp_factor']}\n"
            f"- rx_antenna_rad: {common_params['rx_antenna_rad']}\n"
            f"- Obj_range: {common_params['Obj_range']}"
        )
        plt.figtext(0.73, 0.9, specs_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.4))

    plt.show()

def plot_corner_dist(measurements):
    # Extract Obj_range and RCS_without_const_dBsm from the measurements
    rx_antenna_rad = [entry['azimuth_angle'] for entry in measurements]
    rcs_values = [entry['RCS_with_const_dBsm'] for entry in measurements]

    # Create a scatter plot
    plt.figure(figsize=(12, 6))
    plt.scatter(rx_antenna_rad, rcs_values, color='blue', s=13)
    plt.title('Corner RCS vs azimuth angle')
    plt.xlabel('azimuth angle in degree')
    plt.ylabel('RCS (dBsm)')
    plt.ylim(-40, 40)
    plt.grid(True)

    common_params = extract_common_params(measurements)
    if common_params:
        specs_text = (
            "Specifications:\n"
            f"- Oversamp_factor: {common_params['Oversamp_factor']}\n"
            f"- rx_antenna_rad: {common_params['rx_antenna_rad']}\n"
            f"- Obj_range: {common_params['Obj_range']}"
        )
        plt.figtext(0.73, 0.9, specs_text, fontsize=11, bbox=dict(facecolor='white', alpha=0.4))

    plt.show()


def main():
    Object = "corner".lower()  # sphere, plate, corner
    
    if Object == "plate":
        json_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/LHFT_RCS_Results_Processing/RCS_Plate_vs_azimuth_elevation.json'
        Oversamp_factor = 30  # Specify the desired Oversamp_factor value here
        Obj_range = 2
        measurements = load_data(json_file, Obj_range, Oversamp_factor)
        plot_plate_dist(measurements)

    if Object == "corner":
        Obj_range = 2
        Oversamp_factor = 30
        json_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/LHFT_RCS_Results_Processing/RCS_corner_vs_azimuth_elevation.json'
        measurements = load_data(json_file, Obj_range, Oversamp_factor)
        plot_corner_dist(measurements)

if __name__ == "__main__":
    main()
