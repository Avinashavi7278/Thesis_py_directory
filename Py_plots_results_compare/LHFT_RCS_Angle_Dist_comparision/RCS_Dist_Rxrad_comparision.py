import json
import matplotlib.pyplot as plt

def load_data(json_filename, rx_antenna_rad):
    # Load data from a JSON file and filter based on rx_radius
    with open(json_filename, 'r') as file:
        data = json.load(file)
        measurements = [entry for entry in data['Measurements'] if entry['rx_antenna_rad'] == rx_antenna_rad]
        return measurements

def extract_common_params(measurements):
    # Extract common parameters from the measurements
    if not measurements:
        return None

    common_params = {
        "Oversamp_factor": measurements[0].get("Oversamp_factor", "N/A"),
        "azimuth_angle": measurements[0].get("azimuth_angle", "N/A"),
        "rx_antenna_rad": measurements[0].get("rx_antenna_rad", "N/A")
    }
    return common_params

def plot_plate_dist(measurements):
    # Extract Obj_range and RCS_without_const_dBsm from the measurements
    obj_range = [entry['Obj_range'] for entry in measurements]
    rcs_values = [entry['RCS_without_const_dBsm'] for entry in measurements]

    # Create a scatter plot
    plt.figure(figsize=(10, 5))
    plt.scatter(obj_range, rcs_values, color='blue', s=10)
    plt.title('RCS vs Object Range')
    plt.xlabel('Object Range')
    plt.ylabel('RCS (dBsm)')
    plt.ylim(-10, 20)
    plt.grid(True)

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


def main():
    Object = "plate".lower() # sphere, plate, corner
    rx_antenna_rad = 0.3  #enter in the range from 0.1 to 1m
    if Object == "plate":
        json_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/LHFT_RCS_Angle_Dist_comparision/Plate_RCS_Dist_results.json'
        measurements = load_data(json_file, rx_antenna_rad)
        plot_plate_dist(measurements)

if __name__ == "__main__":
    main()
