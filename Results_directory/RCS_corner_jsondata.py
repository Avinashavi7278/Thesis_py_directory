
import json
import os
import matplotlib.pyplot as plt
import random

def generate_data():
    Obj_range = list(range(5, 105, 5))
    #Calculating Azimuth with respect to sizee of plate, but here I set (azimuth 3.58 constant)
    RCS_dBsm =  [-33.756237692439456, -31.25746296027345, -34.31681216444432, -37.278062873553075, -39.422262266510444, -36.255012424605454, -33.577140839380924, -31.25746296027345, -29.211362062378203, -27.381062439951197, -25.725355033622197, -24.213812598046207]
    # For perfect reflector 
    RCS_dBsm = [1.0510239937051404, -26.4137623, ]
    
    return Obj_range, RCS_dBsm

def save_to_json(Obj_range, RCS_dBsm, filename):

    data = {'Obj_range': Obj_range, 'RCS_dBsm': RCS_dBsm}
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

save_to_json(Obj_range, RCS_dBsm, filename)






# import json
# import os
# import matplotlib.pyplot as plt
# import random

# def generate_data():

#     RCS_range = list(range(5, 105, 5))
#     RCS_dBsm = [-33.756237692439456, -31.25746296027345, -34.31681216444432, -37.278062873553075, -39.422262266510444, -36.255012424605454, -33.577140839380924, -31.25746296027345, -29.211362062378203, -27.381062439951197, -25.725355033622197, -24.213812598046207]
#     return RCS_range, RCS_dBsm

# def save_to_json(RCS_range, RCS_dBsm, filename):

#     data = {'RCS_range': RCS_range, 'RCS_dBsm': RCS_dBsm}
#     with open(filename, 'w') as file:
#         json.dump(data, file, indent=4)
#     print(f"Data saved to {filename}")


# def plot_data(RCS_range, RCS_dBsm):
#     plt.figure(figsize=(10, 5))
#     plt.plot(RCS_range, RCS_dBsm, marker='o', linestyle='-')
#     plt.title('Plot of RCS vs Range')
#     plt.xlabel('Object Range (m)')
#     plt.ylabel('RCS (dBsm)')
#     plt.ylim(-40, 0)  # Set y-axis range from -20 to 5
#     plt.grid(True)
#     plt.show()
# # Generate the data
# RCS_range, RCS_dBsm = generate_data()

# # Filename to save the JSON data
# filename = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Results_directory/Corner_RCS_vs_range.json'
# plot_data(RCS_range, RCS_dBsm)

# # Save the data to a JSON file
# save_to_json(RCS_range, RCS_dBsm, filename)
