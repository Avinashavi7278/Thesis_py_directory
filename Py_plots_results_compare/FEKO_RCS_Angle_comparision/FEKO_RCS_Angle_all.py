import matplotlib.pyplot as plt
from math import radians
import pandas as pd
import numpy as np


#****************************************************Plate************************************************************#

def plate_angle_feko():
    file_MOM = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/FEKO_RCS_Angle_comparision/Plate_MOM60_text.txt'
    # file_Ray = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/FEKO_RCS_Angle_comparision/Plate_Ray60_text.txt'
    # file_UTD = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/FEKO_RCS_Angle_comparision/Plate_UTD60_text.txt'
    file_ana = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/FEKO_RCS_Angle_comparision/rcs_vertical_60Ghz_polarization.csv'
    filedata_MOM = np.loadtxt(file_MOM)
    # filedata_Ray = np.loadtxt(file_Ray)
    # filedata_UTD = np.loadtxt(file_UTD)
    filedata_analytical = pd.read_csv(file_ana)
    #print(filedata_Ray)


    # a = 5 / 100  # 5 cm to meters width
    # b = 5 / 100  # 5 cm to meters hight

    # # Frequency of the incident electromagnetic wave (in Hz)
    # freq = 60e9  # 60 GHz

    # filedata_analytica, rcsdb_v, rcsdb_h, rcsdb_po = rcs_rect_plate(a, b, freq)

    theta = np.deg2rad(filedata_MOM[:, 0])
    r = filedata_MOM[:,1]

    # theta1 = np.deg2rad(filedata_Ray[:, 0])
    # r1 = filedata_Ray[:,1]

    # theta2 = np.deg2rad(filedata_UTD[:, 0])
    # r2 = filedata_UTD[:,1]

    theta3 = np.deg2rad(filedata_analytical.iloc[:, 0])
    r3 = filedata_analytical.iloc[:,1]



    #Ploting figures
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')
    ax.plot(theta, r, label="MOM [dBsm]", color='black', linestyle='dashdot', linewidth=2)

    # # Ploting the data from the second file
    # ax.plot(theta1, r1, label="Ray [dBsm]", color='red',  linestyle='dashed', linewidth=1)

    # # Ploting the data from the third file
    # ax.plot(theta2, r2, label="UTDS [dBsm]", color='green', linestyle='dotted', linewidth=2)

    # Ploting the data from the third file
    ax.plot(theta3, r3, label="Analytical [dBsm]", color='blue', linestyle='-.', linewidth=0.8)

    ax.set_theta_direction(-1)
    #ax.set_rticks(np.linspace(0, np.max(r), 5))  # Adjust radial ticks based on data
    #ax.set_rlabel_position(0)  # Move radial labels away from plotted line
    ax.set_rlim(-60, 20)
    ax.set_rticks([-60, -50, -40, -30, -20, -10, 0, 10, 20]) 
    ax.grid(True)

    ax.set_title("Plate, RCS(dBsm) vs Angle", va='bottom')
    #ax.legend(["MOM"], loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.legend(handles=[
        plt.Line2D([1], [2], linestyle='-.', color='blue', label='Analytical'),
        plt.Line2D([1], [2], linestyle='dotted', color='green', label='UTD'),
        plt.Line2D([1], [2], linestyle='dashed', color='red', label='Ray_Launch'),
        plt.Line2D([1], [2], linestyle='dashdot', color='black', label='MOM')
        ], loc='upper right', bbox_to_anchor=(1.28, 1.1))


    plt.show()

#****************************************************Corner************************************************************#
def corner_angle_feko():
    file_MOM = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/FEKO_RCS_Angle_comparision/Corner_MOM24_r_rotate.txt'
    file_Ray = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/FEKO_RCS_Angle_comparision/Corner_Ray24_r_rotate.txt'
    file_UTD = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/FEKO_RCS_Angle_comparision/Corner_UTD24_r_rotate.txt'

    filedata_MOM = np.loadtxt(file_MOM)
    filedata_Ray = np.loadtxt(file_Ray)
    filedata_UTD = np.loadtxt(file_UTD)
    #print(filedata_MOM)

    theta = np.deg2rad(filedata_MOM[:, 0])
    r = filedata_MOM[:,1]

    theta1 = np.deg2rad(filedata_Ray[:, 0])
    r1 = filedata_Ray[:,1]

    theta2 = np.deg2rad(filedata_UTD[:, 0])
    r2 = filedata_UTD[:,1]


    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')
    ax.plot(theta, r, label="MOM [dBsm]", color='black', linestyle='dashdot', linewidth=2)

    # Plot the data from the second file
    ax.plot(theta1, r1, label="Ray [dBsm]", color='red',  linestyle='dashed', linewidth=1)

    # Plot the data from the third file
    ax.plot(theta2, r2, label="UTDS [dBsm]", color='green', linestyle='dotted', linewidth=2)

    ax.set_theta_direction(-1)
    #ax.set_rticks(np.linspace(0, np.max(r), 5))  # Adjust radial ticks based on data
    # ax.set_rlabel_position(0)  # Move radial labels away from plotted line
    ax.set_rlim(-60, 20)
    ax.set_rticks([-60, -50, -40, -30, -20, -10, 0, 10, 20]) 
    ax.grid(True)

    ax.set_title("Corner, RCS(dBsm) vs Angle", va='bottom')
    #ax.legend(["MOM"], loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.legend(handles=[
        plt.Line2D([1], [2], linestyle='dotted', color='black', label='Mom'),
        plt.Line2D([1], [2], linestyle='dashed', color='red', label='Ray'),
        plt.Line2D([1], [2], linestyle='dashdot', color='green', label='UTD')
        ], loc='upper right', bbox_to_anchor=(1.2, 1.1))


    plt.show()


#****************************************************Sphere************************************************************#
def sphere_angle_feko():
    file_MOM = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/FEKO_RCS_Angle_comparision/Sphere_mom.txt'
    file_Ray = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/FEKO_RCS_Angle_comparision/Sphere_Ray.txt'
    file_UTD = 'D:/FAU Notes/4Master_Thesis/Simulation/Python_Directory/Py_plots_results_compare/FEKO_RCS_Angle_comparision/Sphere_utd.txt'

    filedata_MOM = np.loadtxt(file_MOM)
    filedata_Ray = np.loadtxt(file_Ray)
    filedata_UTD = np.loadtxt(file_UTD)
    #print(filedata_MOM)

    theta = np.deg2rad(filedata_MOM[:, 0])
    r = filedata_MOM[:,1]

    theta1 = np.deg2rad(filedata_Ray[:, 0])
    r1 = filedata_Ray[:,1]

    theta2 = np.deg2rad(filedata_UTD[:, 0])
    r2 = filedata_UTD[:,1]


    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')
    ax.plot(theta, r, label="MOM [dBsm]", color='black', linestyle='dashdot', linewidth=2)

    # Ploting the data from the second file
    ax.plot(theta1, r1, label="Ray [dBsm]", color='red',  linestyle='dashed', linewidth=1)
    ax.plot(theta2, r2, label="UTDS [dBsm]", color='green', linestyle='dotted', linewidth=2)

    ax.set_theta_direction(-1)
    #ax.set_rticks(np.linspace(0, np.max(r), 5))  # Adjust radial ticks based on data
    # ax.set_rlabel_position(0)  # Move radial labels away from plotted line
    ax.set_rlim(-60, 20)
    ax.set_rticks([-60, -50, -40, -30, -20, -10, 0, 10, 20]) 
    ax.grid(True)

    ax.set_title("Sphere, RCS(dBsm) vs Angle", va='bottom')
    #ax.legend(["MOM"], loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.legend(handles=[
        plt.Line2D([1], [2], linestyle='dotted', color='black', label='Mom'),
        plt.Line2D([1], [2], linestyle='dashed', color='red', label='Ray'),
        plt.Line2D([1], [2], linestyle='dashdot', color='green', label='UTD')
        ], loc='upper right', bbox_to_anchor=(1.2, 1.1))


    plt.show()
#***********************************************************************************************************************#


def main():
    Object = "plate".lower()

    if Object == "plate":
        plate_angle_feko()

    if Object == "corner":
        corner_angle_feko()

    if Object == "sphere":
        sphere_angle_feko()

if __name__ == "__main__":
    main()