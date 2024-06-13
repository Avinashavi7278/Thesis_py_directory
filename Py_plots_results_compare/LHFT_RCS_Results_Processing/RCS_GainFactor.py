import json
import matplotlib.pyplot as plt

def load_data(json_filename, Obj_range, Oversamp_factor):
    # Load data from a JSON file and filter based on Oversamp_factor
    with open(json_filename, 'r') as file:
        data = json.load(file)
        measurements = [entry for entry in data['Measurements'] if entry['Oversamp_factor'] == Oversamp_factor and entry['Obj_range'] == Obj_range]
        return measurements

def load_data_all(json_filename, Obj_range):
    # Load data from a JSON file and filter based on Oversamp_factor
    with open(json_filename, 'r') as file:
        data = json.load(file)
        measurements = [entry for entry in data['Measurements'] if entry['Obj_range'] == Obj_range]
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
        # "gain_factor": measurements[0].get("gain_factor", "N/A"),
        "rx_antenna_rad": measurements[0].get("rx_antenna_rad", "N/A")
    }
    return common_params

def plot_plate_Gain_azimuth(measurements):
    # Extract Obj_range and RCS_without_const_dBsm from the measurements
    azimuth_values = [entry['azimuth_angle'] for entry in measurements]
    gain_values = [entry['gain_factor'] for entry in measurements]

    # Create a scatter plot
    plt.figure(figsize=(10, 5))
    plt.scatter(azimuth_values, gain_values, color='blue', s=13)
    plt.title('Plate RCS vs gainfactor')
    plt.xlabel('Azimuth in degree')
    plt.ylabel('Gainfactor')
    plt.ylim(0, 3000)
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


def plot_plate_RCS_azimuth(measurements):
    # Extract Obj_range and RCS_without_const_dBsm from the measurements
    azimuth_values = [entry['azimuth_angle'] for entry in measurements]
    rcs_values = [entry['RCS_with_const_dBsm'] for entry in measurements]

    # Create a scatter plot
    plt.figure(figsize=(10, 5))
    plt.scatter(azimuth_values, rcs_values, color='blue', s=13)
    plt.title('Plate RCS with gain vs Azimuth')
    plt.xlabel('Azimuth in degree')
    plt.ylabel('RCS dBsm')
    plt.ylim(-70, 70)
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



def plot_plate_gain_compare(measurements):
    # Extract data for plotting
    # Extract data for plotting
    azimuth_angles_without_gain = []
    rcs_without_gain = []
    azimuth_angles_with_gain = []
    rcs_with_gain = []

    for entry in measurements:
        if entry['Specification'] == 'RCS_Azimuth_without_gain':
            azimuth_angles_without_gain.append(entry['azimuth_angle'])
            rcs_without_gain.append(entry['RCS_with_const_dBsm'])
        elif entry['Specification'] == 'RCS_Azimuth_with_gain':
            azimuth_angles_with_gain.append(entry['azimuth_angle'])
            rcs_with_gain.append(entry['RCS_with_const_dBsm'])

    # Calculate the offset
    offset = rcs_without_gain[0] - rcs_with_gain[0]
    print(f"the offset is {offset}")
    # Apply the offset to the RCS with gain data
    rcs_with_gain_adjusted = [x + offset for x in rcs_with_gain]


    # Compute the differences between adjusted RCS_with_gain and RCS_without_gain
    rcs_diff_adjusted = [with_gain - without_gain for with_gain, without_gain in zip(rcs_with_gain_adjusted, rcs_without_gain)]

    # Plotting
    plt.figure(figsize=(12, 6))

    plt.plot(azimuth_angles_without_gain, rcs_without_gain, label='RCS Azimuth Without Gain', marker='o')
    plt.plot(azimuth_angles_with_gain, rcs_with_gain_adjusted, label='RCS Azimuth With Gain (Adjusted)', marker='s')
    plt.plot(azimuth_angles_without_gain, rcs_diff_adjusted, label='Difference (With Gain - Without Gain)', marker='x')

    plt.xlabel('Azimuth Angle (degrees)')
    plt.ylabel('RCS (dBsm)')
    plt.title('RCS Azimuth With and Without Gain (Adjusted)')
    plt.legend()
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

def plot_corner_gainfactor(measurements):
    # Extract Obj_range and RCS_without_const_dBsm from the measurements
    azimuth_angle = [entry['azimuth_angle'] for entry in measurements]
    RCS_dbsm = [entry['RCS_with_const_dBsm'] for entry in measurements]

    # Create a scatter plot
    plt.figure(figsize=(12, 6))
    plt.scatter(azimuth_angle, RCS_dbsm, color='blue', s=13)
    plt.title('Corner RCS_dbsm  vs azimuth angle')
    plt.xlabel('azimuth angle in degree')
    plt.ylabel('RCS_dbsm')
    plt.ylim(-35, 35)
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
    Object = "plate_gain_comparision".lower()  # sphere, plate, corner
    
    if Object == "plate_gain":
        json_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/LHFT_RCS_Results_Processing/RCS_Plate_vs_azimuth_elevation.json'
        Oversamp_factor = 30  # Specify the desired Oversamp_factor value here
        Obj_range = 2
        measurements = load_data(json_file, Obj_range, Oversamp_factor)
        plot_plate_Gain_azimuth(measurements)

    if Object == "plate_rcs_azimuth":
        json_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/LHFT_RCS_Results_Processing/RCS_Plate_vs_azimuth_elevation.json'
        # Oversamp_factor = 60  # Specify the desired Oversamp_factor value here
        Obj_range = 2
        measurements = load_data_all(json_file, Obj_range)
        plot_plate_RCS_azimuth(measurements)


    if Object == "corner":
        Obj_range = 2
        Oversamp_factor = 30
        json_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/LHFT_RCS_Results_Processing/RCS_corner_vs_azimuth_elevation.json'
        measurements = load_data(json_file, Obj_range, Oversamp_factor)
        plot_corner_gainfactor(measurements)

    if Object == "plate_gain_comparision":
        Obj_range = 2
        Oversamp_factor = 60
        json_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/LHFT_RCS_Results_Processing/RCS_plate_gainfactor_data_new.json'
        measurements = load_data(json_file, Obj_range, Oversamp_factor)
        plot_plate_gain_compare(measurements)

if __name__ == "__main__":
    main()


# Obj_range, mesh_angle_r, mesh_angle_up, RCS_with_const_dBsm, gain_factor, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle, elevation_angle,diff_const
# Obj_range, mesh_angle_r, mesh_angle_up, RCS_without_const_dBsm, gain_factor, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle, diff_const