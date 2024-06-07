import matplotlib.pyplot as plt
import numpy as np


def plot_plate_size():
    # over sampling factor 60, Obj_range 2, diffusion 0.0
    plate_size = np.array([5, 7, 9, 11, 13, 15])
    FEKO_plate = np.array([ 4.9863, 10.8286, 15.1930, 18.6781, 21.5795, 24.0650])
    Analy_solution = np.array([4.97, 10.82, 15.18, 18.67, 21.57, 24.06])
    LHFT_plate = np.array([ 4.9869, 10.8229, 15.1840, 18.6746, 21.5743, 24.0593])



    # Plot all four arrays in one single plot
    plt.plot(plate_size, FEKO_plate, label='plate in FEKO',  linestyle='-', marker='o', linewidth=3)
    plt.plot(plate_size, LHFT_plate, label='plate in LHFT (RCS_const = 0.3164)',  linestyle='-.', marker='x', linewidth=2)
    plt.plot(plate_size, Analy_solution, label='plate analytical solution',  linestyle=':', marker='x', linewidth=2)
    


    plt.title('RCS vs plate with different size')
    plt.xlabel('Plate size in cm')
    plt.ylabel('RCS in dBsm')
    plt.xticks(np.arange(4, 16, 1))

    # Add a legend
    plt.legend()
    plt.grid(True)
    # Show the plot
    plt.show()

def plot_corner_size():
    # over sampling factor 60, diffusion 0.0
    plate_size = np.array([5, 7, 9, 11, 13, 15])
    FEKO_corner = np.array([-0.7554, 5.1019, 9.4586, 12.9435, 15.8628, 18.3327])
    LHFT_corner = np.array([ -0.7556, 5.0809, 9.4334, 12.9000, 15.7728, 18.2362])#5m range
    LHFT_corner_using_ConstPlate1 = np.array([-1.9271, 3.8698, 8.1920, 11.6437, 14.5103, 16.9548])#2m range
    # LHFT_corner_using_ConstPlate2 = np.array([-2.0443, 3.7922, 8.1447, 11.6113, 14.4841, 16.9474])#5m range
    # Plot all four arrays in one single plot
    plt.plot(plate_size, FEKO_corner, label='Corner in FEKO', linestyle='-', marker='o', linewidth=3)
    plt.plot(plate_size, LHFT_corner, label='Corner in LHFT (RCS_const = 0.42465)', linestyle='--', marker='x', linewidth=2)
    plt.plot(plate_size, LHFT_corner_using_ConstPlate1, label='Corner in LHFT (RCS_const = 0.3164)')

    plt.title('RCS vs corner with different size')
    plt.xlabel('Corner size in cm')
    plt.ylabel('RCS in dBsm')
    plt.xticks(np.arange(4, 16, 1))

    # Add a legend
    plt.legend()
    # Show the plot
    plt.grid(True)
    plt.show()


def plot_sphere_size():
    # over sampling factor 60, Obj_range 2, diffusion 0.0
    plate_size = np.array([5, 7, 9, 11, 13, 15])
    FEKO_sphere = np.array([ -20.5327, -17.6635, -15.4889, -13.9, -12.5, -11.5])
    LHFT_sphere = np.array([ -20.5327, -14.6727, -10.1210, -6.5294, -3.3758, -0.8055])

    # Plot all four arrays in one single plot
    plt.plot(plate_size, FEKO_sphere, label='sphere in FEKO')
    plt.plot(plate_size, LHFT_sphere, label='sphere in LHFT')

    plt.title('RCS vs plate with different size')
    plt.xlabel('Sphere size in cm')
    plt.ylabel('RCS in dBsm')
    plt.xticks(np.arange(4, 16, 1))

    # Add a legend
    plt.legend()
    # Show the plot
    plt.grid(True)
    plt.show()

def main():

    Object = "plate".lower() #sphere, plate, corner

    if Object == "plate":
        plot_plate_size()

    if Object == "corner":
        plot_corner_size()

    if Object == "sphere":
        plot_sphere_size()

if __name__ == "__main__":
    main()
