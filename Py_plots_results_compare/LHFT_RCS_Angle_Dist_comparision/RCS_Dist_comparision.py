import json
import matplotlib.pyplot as plt

def load_data(json_filename):
    # Load data from a JSON file
    with open(json_filename, 'r') as file:
        data = json.load(file)
        return data['Measurements']

def plot_data(measurements):
    # Extract Obj_range and RCS_without_const_dBsm from the measurements
    obj_range = [entry['Obj_range'] for entry in measurements]
    rcs_values = [entry['RCS_without_const_dBsm'] for entry in measurements]

    # Create a scatter plot
    plt.figure(figsize=(10, 5))
    plt.scatter(obj_range, rcs_values, linestyle='-', color='blue')
    plt.title('RCS vs Object Range')
    plt.xlabel('Object Range')
    plt.ylabel('RCS (dBsm)')
    plt.ylim(-10, 20)
    plt.grid(True)
    plt.show()

def main():
    json_filename = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare\LHFT_RCS_Angle_Dist_comparision/Plate_RCS_Dist_results.json'
    measurements = load_data(json_filename)
    plot_data(measurements)

if __name__ == "__main__":
    main()
