import json
import matplotlib.pyplot as plt
import pandas as pd

def load_data_detailed(json_filename, rx_antenna_rad, Oversamp_factor):
    # Load data from a JSON file and filter based on rx_radius
    with open(json_filename, 'r') as file:
        data = json.load(file)
        measurements = [entry for entry in data['Measurements'] if entry['rx_antenna_rad'] == rx_antenna_rad and entry['Oversamp_factor'] == Oversamp_factor]
        return measurements

def load_data(json_filename):
    # Load data from a JSON file and filter based on rx_radius
    with open(json_filename, 'r') as file:
        data = json.load(file)
        measurements = [entry for entry in data['Measurements'] ]
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
    Obj_range = [entry['Obj_range'] for entry in measurements]
    rcs_values = [entry['RCS_without_const_dBsm'] for entry in measurements]

    # Create a scatter plot
    plt.figure(figsize=(10, 5))
    plt.scatter(Obj_range, rcs_values, linestyle='-', color='blue', s=10)
    plt.title('Plate RCS vs Object Range')
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

def plot_plate_diff(measurements):
    # Extract diff_const and RCS_without_const_dBsm from the measurements
    diff_const = [entry['diff_const'] for entry in measurements]
    rcs_values = [entry['RCS_without_const_dBsm'] for entry in measurements]

    # Create a subplot layout with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Create the scatter plot in the first subplot
    ax1.scatter(diff_const, rcs_values, color='blue', s=10)
    ax1.set_title('Plate RCS vs Object Range')
    ax1.set_xlabel('Diffusion Range')
    ax1.set_ylabel('RCS (dBsm)')
    ax1.set_ylim(-10, 20)
    ax1.grid(True)

    common_params = extract_common_params(measurements)
    if common_params:
        # Create a table with the specifications in the second subplot
        data = {
            "Diffusion": ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"],
            "Rotation (degrees)": ["10", "13", "17", "22", "30", "42", "89", "89"]
        }
        df = pd.DataFrame(data)
        ax2.axis('tight')
        ax2.axis('off')
        table = ax2.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(0.8, 0.8)

    plt.tight_layout()
    plt.show()
# def plot_plate_diff(measurements):
#     # Extract Obj_range and RCS_without_const_dBsm from the measurements
#     diff_const = [entry['diff_const'] for entry in measurements]
#     rcs_values = [entry['RCS_without_const_dBsm'] for entry in measurements]

#     # Create a scatter plot
#     plt.figure(figsize=(10, 5))
#     plt.scatter(diff_const, rcs_values, linestyle='-', color='blue', s=10)
#     plt.title('Plate RCS vs Object Range')
#     plt.xlabel('Object Range')
#     plt.ylabel('RCS (dBsm)')
#     plt.ylim(-10, 20)
#     plt.grid(True)

#     common_params = extract_common_params(measurements)
#     if common_params:
#         specs_text = (
#             "Specifications:\n"
#             "- Exact edge of plate is 90 degree"
#             "- For 0 diffusion : 10 degree rotation"
#             "- For 0.1 diffusion : 13 degree rotation"
#             "- For 0.2 diffusion : 17 degree rotation"
#             "- For 0.3 diffusion : 22 degree rotation"
#             "- For 0.4 diffusion : 30 degree rotation"
#             "- For 0.5 diffusion : 42 degree rotation"
#             "- For 0.6 diffusion : 89 degree rotation"
#             "- For 0.7 diffusion : 89 degree rotation"
#         )
#         plt.figtext(0.73, 0.9, specs_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.4))
#     plt.show()

def plot_sphere_dist(measurements):
    # Extract Obj_range and RCS_without_const_dBsm from the measurements
    obj_range = [entry['Obj_range'] for entry in measurements]
    rcs_values = [entry['RCS_without_const_dBsm'] for entry in measurements]

    # Create a scatter plot
    plt.figure(figsize=(10, 5))
    plt.scatter(obj_range, rcs_values, linestyle='-', color='blue', s=10)
    plt.title('Sphere RCS vs Range')
    plt.xlabel('Object Range')
    plt.ylabel('RCS (dBsm)')
    plt.ylim(-90, 0)
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


def plot_corner_dist(measurements):
    # Extract Obj_range and RCS_without_const_dBsm from the measurements
    obj_range = [entry['Obj_range'] for entry in measurements]
    rcs_values = [entry['RCS_without_const_dBsm'] for entry in measurements]

    # Create a scatter plot
    plt.figure(figsize=(10, 5))
    plt.scatter(obj_range, rcs_values, linestyle='-', color='blue', s=8)
    plt.title('Corner RCS vs Range')
    plt.xlabel('Object Range')
    plt.ylabel('RCS (dBsm)')
    plt.ylim(-20, 20)
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
    Object = "corner".lower() #sphere, plate, corner
    rx_antenna_rad = 0.3
    Oversamp_factor = 110 #choose 30, 60 for diffusion, but for better results choose 90 only with rx_antenna_rad = 0.3, 110 for corner

    if Object == "sphere":
        json_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/LHFT_RCS_Results_Processing/Sphere_RCS_Dist_results.json'
        measurements = load_data(json_file)
        plot_sphere_dist(measurements)

    if Object == "corner":
        json_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/LHFT_RCS_Results_Processing/Corner_RCS_Dist_results.json'
        measurements = load_data_detailed(json_file, rx_antenna_rad, Oversamp_factor)
        plot_corner_dist(measurements)

    if Object == "plate":
        json_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/LHFT_RCS_Results_Processing/Plate_RCS_Dist_results.json'
        measurements = load_data_detailed(json_file, rx_antenna_rad, Oversamp_factor)
        plot_plate_dist(measurements)

    if Object == "plate_diff":
        Oversamp_factor = 60
        json_file = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/LHFT_RCS_Results_Processing/Plate_RCS_Dist_results.json'
        measurements = load_data_detailed(json_file, rx_antenna_rad, Oversamp_factor)
        plot_plate_diff(measurements)

if __name__ == "__main__":
    main()
