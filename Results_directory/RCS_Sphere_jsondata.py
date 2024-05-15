import json
import os
import matplotlib.pyplot as plt
import random

def generate_data():

    RCS_range = list(range(5, 105, 5))
    RCS_dBsm = [-20.973480843879422, -21.129434419684824, -21.24041779420288, -21.112104721057072, -20.837034647598124, -20.89714485039263, -20.9826133120092, -21.49640931794149, -21.65094222664547, -22.05698378994636, -22.236683850730266, -22.207175825521276, -22.122026205404627, -22.51267656495276, -22.700677457374386, -22.45763822734628, -23.720387821820918, -22.972133601159864, -22.811251087786346, -22.480769771345123]
    Angle = list(range(10, 370, 10))
    RCS_vs_angle = [-27.1985847446308, -28.02379698790855, -29.423254947777636, -31.54539446818918, -34.597670874408244, -38.95857374131979, -45.54752864008199, -57.332108366253365]
    return RCS_range, RCS_dBsm, Angle, RCS_vs_angle

def save_to_json(RCS_range, RCS_dBsm, Angle, RCS_vs_angle, filename):

    data = {'RCS_range': RCS_range, 'RCS_dBsm': RCS_dBsm, 'Angle': Angle, 'RCS_vs_angle':RCS_vs_angle}
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to {filename}")


def plot_data(RCS_range, RCS_dBsm):

    plt.figure(figsize=(10, 5))
    plt.plot(RCS_range, RCS_dBsm, marker='o', linestyle='-')
    plt.title('Plot of RCS vs Range')
    plt.xlabel('Object Range (m)')
    plt.ylabel('RCS (dBsm)')
    plt.ylim(-40, -10)  # Set y-axis range from -20 to 5
    plt.grid(True)
    plt.show()
# Generate the data
RCS_range, RCS_dBsm , Angle , RCS_vs_angle = generate_data()

# Filename to save the JSON data
filename = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Results_directory/Sphere_RCS_vs_range.json'
# plot_data(RCS_range, RCS_dBsm)

# Save the data to a JSON file
save_to_json(RCS_range, RCS_dBsm, Angle, RCS_vs_angle, filename)
