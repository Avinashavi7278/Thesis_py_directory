import json
import os
import matplotlib.pyplot as plt
import random

def generate_data():
    Obj_range = list(range(5, 105, 5))
    #Calculating Azimuth with respect to sizee of plate, but here I set (azimuth 3.58 constant)
    RCS_dBsm = [4.765926, 4.7163191, 4.6601944, 4.5875520, 4.63017614, 4.66076702, 4.6222559, 4.5633952, 4.5541903, 4.5821611, 4.5379130203703895, 4.650454642952465, 4.7477424883326975, 4.7835699718030735, 4.931342247417847, 5.123455062886997, 5.222772805023466, 4.916900015364459, 4.871783395941743, 4.965193410199616]
    # Changing the azimuth over the range 
    # RCS_dBsm = [-38.45488, -26.4137623. ]
    
    return Obj_range, RCS_dBsm

def save_to_json(Obj_range, RCS_dBsm, freq,filename):

    data = {'Obj_range': Obj_range, 'RCS_dBsm': RCS_dBsm, 'freq': freq}
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to {filename}")

def plot_data(Obj_range, RCS_dBsm):

    plt.figure(figsize=(10, 5))
    plt.plot(Obj_range, RCS_dBsm, marker='o', linestyle='-')
    plt.title('Plot of RCS vs Range')
    plt.xlabel('Object Range (m)')
    plt.ylabel('RCS (dBsm)')
    plt.ylim(-10, 10)  # Set y-axis range from -20 to 5
    plt.grid(True)
    plt.show()

# Generate the data
Obj_range, RCS_dBsm = generate_data()

# Filename to save the JSON data
filename = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Results_directory/RCS_Plate_vs_range.json'

# Plot the data
plot_data(Obj_range, RCS_dBsm)
# Save the data to a JSON file
freq =10
save_to_json(Obj_range, RCS_dBsm, freq, filename)
