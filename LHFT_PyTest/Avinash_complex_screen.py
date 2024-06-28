28.06.2024

import os
import sys
python_file_directory = os.path.dirname(os.path.abspath(__file__))
upper_directory = python_file_directory + "/../"
sys.path.append(upper_directory)
import json
from scipy.integrate import dblquad
from radar_ray_python.Persistence import save_radar_measurement_as_binary
from radar_ray_python.Renderer import RenderMode, Renderer, RayChannelInfo
from radar_ray_python.RxAntenna import RxAntenna
from radar_ray_python.TxAntenna import TxAntenna
from radar_ray_python.RadiationPattern import *
from radar_ray_python.Material import MaterialDielectric, MaterialLambertian, MaterialMetal, MaterialMixed
import radar_ray_python as raray
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
import matplotlib as mpl
mpl.rc('font',family='Times New Roman')

import scipy.io as sio

from radar_ray_python.data import *
from radar_ray_python.Persistence import load_mesh_normal_from_obj

from Avinash_RCS import calculate_azimuth_angle,save_to_json, calculate_elevation_angle, load_antennas_for_imaging_iwr6843AOP, load_Object

def load_Objects(renderer, material_dir, obj_filename, mesh_angle_r, mesh_angle_up, diff_const):
    mesh_list, obj_mat_list = load_mesh_normal_from_obj(obj_filename, material_dir)
    for i, obj_mat in enumerate(obj_mat_list):
        mesh = mesh_list[i]
        if "metal" in obj_mat.name.lower():
            mesh_mat = MaterialMixed(obj_mat.diffuse, diff_const)
            mesh.set_id(0)
            #mesh_mat = MaterialMetal(obj_mat.diffuse, 0.1)
        elif "ground" in obj_mat.name.lower():
            mesh_mat = MaterialMixed(obj_mat.diffuse, 0.1)
            mesh.set_id(1)
        else:
            mesh_mat = MaterialMixed(obj_mat.diffuse, 0.1)


        mesh.set_material(mesh_mat)
        mesh.rotate([1.0, 0.0, 0.0],np.deg2rad(mesh_angle_up))
        mesh.rotate([0.0, 0.0, 1.0],np.deg2rad(mesh_angle_r))
        renderer.add_geometry_object(mesh)
    print("Mesh loading completed")




def simulate_complex_scene(image_width, image_height, Oversamp_factor, Wavelength, 
                      rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle, elevation_angle,
                      mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type, diff_const, RCS_const):
   renderer = Renderer()
   render_mode = getattr(RenderMode, render_mode_type)
   print("function call successful")
   # load mesh from outside

   script_directory = os.path.dirname(__file__)
   content_directory = os.path.join(script_directory, "../example-files/Avinash_RCS/Complex_scene/")
   obj_filename = os.path.join(content_directory, "Complex_sceen_without_rings.obj")#Plate_high_roughness_metal, Plate_high_roughness_flipped
   #print("start loading campus scene")
   load_Objects(renderer, content_directory, obj_filename, mesh_angle_r, mesh_angle_up, diff_const)

   radiation_pattern_filename = os.path.join(script_directory, "../example-files/Radiation_pattern_new.txt")
   #radiation_pattern_filename = os.path.join(script_directory, "../example-files/test_radiation_pattern.txt")
   radiation_pattern, phi_axis, theta_axis = load_radiation_pattern_from_cst(radiation_pattern_filename)
   # plot_radiation_pattern_3d(radiation_pattern, phi_axis, theta_axis, subsampling_factor=8)
   # plt.show()


   if render_mode == RenderMode.RENDER_MODE_GRAPHICS:
      # From Vector (from_vec) - This is the position of the camera itself. You need to place it at a suitable distance to view the plate clearly. Assuming the plate is at the origin and the camera needs to be positioned directly in front of it, you could set the camera at a position along the z-axis.
      # At Vector (at_vec) - This vector points to where the camera is looking at. Since the plate is at the origin and we want the camera to focus there
      # Up Vector (up_vec) - This defines the upward direction relative to the camera's point of view. Since the plate is rotated 90 degrees, and assuming the rotation is about the y-axis making the top of the plate align with the x-axis, you would typically want the up vector to align with the y-axis to keep the camera's view upright
      # set_camera ([move camera with x for f and b, y to l and r, z for up and d ],[Looking camera at x f and b, y l and r, z up and down(our case looking origin000)],[up_vector])
      renderer.set_camera(From_vector, look_at_front, vec_up)
      renderer.set_render_mode(render_mode)
      renderer.set_image_size(image_width, image_height)
      renderer.set_oversampling_factor(Oversamp_factor)
      renderer.initialize()
      renderer.render()
      image = renderer.get_image()
      image_array = Image.fromarray(image)
      render_directory = os.path.join(script_directory, "../example-scripts/Avinash_render_images/")
      print("Image stored in RCS_complex_sceen")
      # image_array.save("RCS_corner_multipath.png")
      # Ensure the directory exists
      os.makedirs(render_directory, exist_ok=True)

      # Define the full path to the image file
      image_path = os.path.join(render_directory, "RCS_complex_sceen.png")

      # Save the image to the specified directory
      image_array.save(image_path)

   elif render_mode == RenderMode.RENDER_MODE_RAYTARGET_COMP:
      # This mode supports accurate Doppler and Radiation-Patterns
      
      tx_antennas, rx_antennas = load_antennas_for_imaging_iwr6843AOP(From_vector, look_at_front, vec_up,
                                         radiation_pattern, phi_axis, theta_axis, Wavelength, rx_antenna_rad, azimuth_angle, elevation_angle)
      # tx_antennas, rx_antennas = set_antennas_quad_digimmic_3_16(
      #    None, antenna_offset, rx_radius=0.5, cone_angle_deg=cone_angle_deg, 
      #    radiation_pattern=radiation_pattern, phi_axis=phi_axis, theta_axis=theta_axis, look_dir=look_dir, up_vector=up_vec)

      print("Loading antenna for imaging completed")
      renderer.add_rx_antenna(rx_antennas[0])
      renderer.add_tx_antenna(tx_antennas[0])
      renderer.set_ray_depth(4)
      renderer.set_number_sequences(10)
      renderer.set_render_mode(render_mode)
      renderer.set_image_size(image_width, image_height)
      renderer.set_oversampling_factor(Oversamp_factor)
      time_start = time.time()
      renderer.initialize()
      renderer.render()
      print("Render complete")
      renderer.create_channel_info()
      time_end = time.time()
      print(f"ray tracing simulation took: {time_end - time_start:.2f} seconds")
      ray_channel_info = renderer.get_channel_info()
      time_start = time.time()
      trace_data = create_trace_data_from_ray_channel_info(ray_channel_info, tx_antennas, rx_antennas, apply_radiation_pattern=True, all_radiation_patterns_equal=True)
      time_end = time.time()
      print(f"creating trace data took: {time_end - time_start:.2f} seconds")

      number_rays = trace_data.traces_dict[0,0].shape[1]
      print(f"received {number_rays} rays")

      Pt = (image_width*image_height)*Oversamp_factor
      Power_ratio = (number_rays/Pt)**2
      RCS_without_const = Power_ratio*(4*np.pi*(Obj_range**2))**2
      RCS_with_const = RCS_without_const * RCS_const

      print(f"azimuth in degree  : {azimuth_angle:.8f} ")
      print(f"elevation in degree  : {elevation_angle:.8f} ")

      azimuth_radi = azimuth_angle * (np.pi / 180)
      elevation_radi = elevation_angle * (np.pi / 180)

      print(f"azimuth in radians  : {azimuth_radi:.8f} ")
      print(f"elevation in radians  : {elevation_radi:.8f} ")
      # For the effective area, performing double integral over azimuth and elevation, which provides effective area in the omni directional sphere 
      effective_area, error = dblquad(lambda theta, phi: np.sin(azimuth_radi),
                                0, 2 * azimuth_radi,  # Outer integral bounds (azimuth)
                                lambda x: 0, lambda x: 2 * elevation_radi)  # Inner integral bounds (elevation)

      print(f"effective_area is: {effective_area:.8f}")

      gain_factor = (4 * np.pi) / effective_area
      print(f"gain_factor : {gain_factor:.8f}")
      gain_factor_dBsm = 20 * np.log10(gain_factor)

      print(f"gain_factor_dBsm : {gain_factor_dBsm:.8f}")

      # RCS_with_const_gain = RCS_with_const / gain_factor

      RCS_with_const_dBsm = 10 * np.log10(RCS_with_const) 

      RCS_with_const_gain_dbsm = RCS_with_const_dBsm - gain_factor_dBsm

      print(f" RCS in dBsm {RCS_with_const_gain_dbsm}") 

   
      return RCS_with_const_gain_dbsm, gain_factor_dBsm

def main():
    #Parameters for incident rays 
    image_width = 1200
    image_height = 600
    Oversamp_factor = 60
    Wavelength = 0.005

    # variables to change the range and theta
    Obj_range = 4

    iterations = 1
    range_azimuth = 2
    rx_antenna_rad = 0.3


    look_at_front =  np.array([0.0, 0, 0.0])
    vec_up = np.array([0.0, 0.0, 1.0]) 

    RCS_const = 0
    render_mode_type = "RENDER_MODE_GRAPHICS"   #RENDER_MODE_RAYTARGET_COMP,  RENDER_MODE_GRAPHICS
    obj_width = 0.05
    obj_length = 0.05
    mesh_angle_r = 90
    mesh_angle_up = 90
    diff_const = 0
    # rx_antenna_rad = 0.0
    RCS_const = 0.3164 
    specification= "RCS_with_gain_diff_0_1"


    azimuth_angle = calculate_azimuth_angle(obj_width, range_azimuth)
    elevation_angle = calculate_elevation_angle(obj_length, range_azimuth)


    # azimuth_angle_rad = np.deg2rad(azimuth_angle)
    # RCS_without_const_dBsm = simulate_plate(image_width, image_height, Oversamp_factor, Wavelength, 
    #                 rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle,
    #                 mesh_angle_r, Obj_range, render_mode_type)
    # azimuth_angle_rad = np.deg2rad(azimuth_angle)

    script_directory = os.path.dirname(__file__)
    jsonfile_directory = os.path.join(script_directory, "../example-scripts/Avinash_json_files/Plate_Json/")
    json_filename = os.path.join(jsonfile_directory, "RCS_plate_gainfactor_data_new.json")
    for i in range(iterations):
        # azimuth_angle +=  1
        # elevation_angle += 1
        From_vector = np.array([10, 0.0, 2])
        look_at_front =  np.array([0.0, 0, 2.5])
        vec_up = np.array([0.0, 0.0, 1.0]) 
        print(f"The   azimuth_angle is {azimuth_angle:.2f}")
        print(f"The   elevation_angle is {elevation_angle:.2f}")
        print(f"The mesh_angle_up angle is set to {mesh_angle_up:.2f} degrees.")
        RCS_with_const_dBsm, gain_factor = simulate_complex_scene(image_width, image_height, Oversamp_factor, Wavelength, 
                    rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle, elevation_angle,
                    mesh_angle_r, mesh_angle_up,  Obj_range, render_mode_type, diff_const, RCS_const)
        time.sleep(3)
        # save_to_json(Obj_range, mesh_angle_r, mesh_angle_up, RCS_with_const_dBsm, gain_factor, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle, elevation_angle, diff_const, specification)


         
if __name__ =="__main__":
    main()