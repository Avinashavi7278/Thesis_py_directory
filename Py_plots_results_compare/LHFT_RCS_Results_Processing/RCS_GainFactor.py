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
        elif entry['Specification'] == 'RCS vs Azimuth with gain (20log)':
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

    plt.plot(azimuth_angles_without_gain, rcs_without_gain, label='RCS Azimuth Without Gain', marker='o', linewidth=0.5, markersize=4)
    plt.plot(azimuth_angles_with_gain, rcs_with_gain_adjusted, label='RCS Azimuth With Gain', marker='s', linewidth=0.5, markersize=3)


    plt.xlabel('Azimuth Angle (degrees)')
    plt.ylabel('RCS (dBsm)')
    plt.title('RCS Azimuth With and Without Gain (Adjusted)')
    plt.legend()
    plt.ylim(-60, 30)
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


def plot_plate_gainfactor_3d_plot(measurements):
     # Extract data for plotting
    azimuth_angles = []
    elevation_angles = []
    rcs_values = []
    colors = []

    for entry in measurements:
        if entry['Specification'] == 'RCS vs Azimuth with gain (20log)':
            azimuth_angles.append(entry['azimuth_angle'])
            elevation_angles.append(entry['elevation_angle'])
            rcs_values.append(entry['RCS_with_const_dBsm'])
            colors.append('blue')
        elif entry['Specification'] == 'RCS_with_gain_only_elevation':
            azimuth_angles.append(entry['azimuth_angle'])
            elevation_angles.append(entry['elevation_angle'])
            rcs_values.append(entry['RCS_with_const_dBsm'])
            colors.append('green')
        elif entry['Specification'] == 'RCS_with_gain_only_azimuth':
            azimuth_angles.append(entry['azimuth_angle'])
            elevation_angles.append(entry['elevation_angle'])
            rcs_values.append(entry['RCS_with_const_dBsm'])
            colors.append('yellow')

    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the data
    scatter = ax.scatter(azimuth_angles, elevation_angles, rcs_values, c=colors, marker='o')

    ax.set_xlabel('Azimuth Angle (degrees)')
    ax.set_ylabel('Elevation Angle (degrees)')
    ax.set_zlabel('RCS (dBsm)')
    ax.set_title('RCS with Gain vs Azimuth and Elevation')
    ax.set_zlim(-30, -90)

    # Create a legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='RCS with gain vs constant azimuth elevation change', markerfacecolor='blue', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='RCS with gain only elevation', markerfacecolor='green', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='RCS with gain only azimuth', markerfacecolor='yellow', markersize=10)]
    ax.legend(handles=legend_elements)

    common_params = extract_common_params(measurements)
    if common_params:
        specs_text = (
            "Specifications:\n"
            f"- Oversamp_factor: {common_params['Oversamp_factor']}\n"
            f"- rx_antenna_rad: {common_params['rx_antenna_rad']}\n"
            f"- Obj_range: {common_params['Obj_range']}\n"
        )
        plt.figtext(0.2, 0.2, specs_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.4))

    plt.show()


def plot_plate_gain_azimuth(measurements):
    # Extract data for plotting
    azimuth_angles_with_gain = []
    rcs_with_gain_elevation = []

    for entry in measurements:
        if entry['Specification'] == 'RCS_with_gain_only_azimuth':
            azimuth_angles_with_gain.append(entry['azimuth_angle'])
            rcs_with_gain_elevation.append(entry['RCS_with_const_dBsm'])

    # Plotting
    plt.figure(figsize=(12, 6))

    plt.plot(azimuth_angles_with_gain, rcs_with_gain_elevation, label='RCS Azimuth With Gain', marker='s', linewidth=0.5, markersize=3)

    plt.xlabel('Azimuth Angle (degrees)')
    plt.ylabel('RCS (dBsm)')
    plt.title('RCS Azimuth With Gain (Adjusted)')
    plt.legend()
    plt.ylim(-80, -40)
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



def plot_plate_gain_elevation(measurements):
    # Extract data for plotting
    azimuth_angles_with_gain = []
    rcs_with_gain_elevation = []

    for entry in measurements:
        if entry['Specification'] == 'RCS_with_gain_only_elevation':
            azimuth_angles_with_gain.append(entry['elevation_angle'])
            rcs_with_gain_elevation.append(entry['RCS_with_const_dBsm'])

    # Plotting
    plt.figure(figsize=(12, 6))

    plt.plot(azimuth_angles_with_gain, rcs_with_gain_elevation, label='RCS Elevation With Gain', marker='s', linewidth=0.5, markersize=3)

    plt.xlabel('Elevation Angle (degrees)')
    plt.ylabel('RCS (dBsm)')
    plt.title('RCS Elevation With Gain (Adjusted)')
    plt.legend()
    plt.ylim(-80, -40)
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

def plot_plate_gain_diff(measurements):
    # Initialize data containers
    data = {
        'RCS_with_gain_diff_0_25': [],
        'RCS_with_gain_diff_0_5': [],
        'RCS_with_gain_diff_0_75': [],
        'RCS_with_gain_diff_0_1': [],
        'RCS vs Azimuth with gain (20log)': []
    }
    
    # Extract data for plotting
    for entry in measurements:
        spec = entry['Specification']
        if spec in data:
            data[spec].append((entry['azimuth_angle'], entry['RCS_with_const_dBsm']))

    # Sort data by azimuth_angle for each specification
    for spec in data:
        data[spec].sort()

    # Unpack data for plotting
    azimuth_angles = {spec: [x[0] for x in data[spec]] for spec in data}
    rcs_values = {spec: [x[1] for x in data[spec]] for spec in data}

    # Plotting
    plt.figure(figsize=(12, 6))

    plt.plot(azimuth_angles['RCS vs Azimuth with gain (20log)'], rcs_values['RCS vs Azimuth with gain (20log)'], label='RCS with gain diff 0', marker='s', linewidth=0.5, markersize=3)
    plt.plot(azimuth_angles['RCS_with_gain_diff_0_25'], rcs_values['RCS_with_gain_diff_0_25'], label='RCS with gain diff 0.25', marker='s', linewidth=0.5, markersize=3)
    plt.plot(azimuth_angles['RCS_with_gain_diff_0_5'], rcs_values['RCS_with_gain_diff_0_5'], label='RCS with gain diff 0.5', marker='s', linewidth=0.5, markersize=3)
    plt.plot(azimuth_angles['RCS_with_gain_diff_0_75'], rcs_values['RCS_with_gain_diff_0_75'], label='RCS with gain diff 0.75', marker='s', linewidth=0.5, markersize=3)
    plt.plot(azimuth_angles['RCS_with_gain_diff_0_1'], rcs_values['RCS_with_gain_diff_0_1'], label='RCS with gain diff 1', marker='s', linewidth=0.5, markersize=3)

    plt.xlabel('Azimuth and elevation Angle (degrees)')
    plt.ylabel('RCS (dBsm)')
    plt.title('RCS With Gain (Adjusted) for different diffusion')
    plt.legend()
    plt.ylim(-120, -40)
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

def main():
    Object = "rcs_gain_3d_plot".lower()  # sphere, plate, corner
    
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


    if Object == "rcs_gain_3d_plot":
        Obj_range = 2
        Oversamp_factor = 60
        json_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/LHFT_RCS_Results_Processing/RCS_plate_gainfactor_data_new.json'
        measurements = load_data(json_file, Obj_range, Oversamp_factor)
        plot_plate_gainfactor_3d_plot(measurements)

    if Object == "plate_gain_comparision":
        Obj_range = 2
        Oversamp_factor = 60
        json_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/LHFT_RCS_Results_Processing/RCS_plate_gainfactor_data_new.json'
        measurements = load_data(json_file, Obj_range, Oversamp_factor)
        plot_plate_gain_compare(measurements)

    if Object == "plate_gain_azimuth":
        Obj_range = 2
        Oversamp_factor = 60
        json_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/LHFT_RCS_Results_Processing/RCS_plate_gainfactor_data_new.json'
        measurements = load_data(json_file, Obj_range, Oversamp_factor)
        plot_plate_gain_azimuth(measurements)

    if Object == "plate_gain_elevation":
        Obj_range = 2
        Oversamp_factor = 60
        json_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/LHFT_RCS_Results_Processing/RCS_plate_gainfactor_data_new.json'
        measurements = load_data(json_file, Obj_range, Oversamp_factor)
        plot_plate_gain_elevation(measurements)

    if Object == "plate_gain_diff":
        Obj_range = 2
        Oversamp_factor = 60
        json_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/LHFT_RCS_Results_Processing/RCS_plate_gainfactor_data_new.json'
        measurements = load_data(json_file, Obj_range, Oversamp_factor)
        plot_plate_gain_diff(measurements)

if __name__ == "__main__":
    main()


# Obj_range, mesh_angle_r, mesh_angle_up, RCS_with_const_dBsm, gain_factor, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle, elevation_angle,diff_const
# Obj_range, mesh_angle_r, mesh_angle_up, RCS_without_const_dBsm, gain_factor, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle, diff_const