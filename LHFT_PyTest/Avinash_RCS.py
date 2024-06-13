23/5/2024

import os
import sys
python_file_directory = os.path.dirname(os.path.abspath(__file__))
upper_directory = python_file_directory + "/../"
sys.path.append(upper_directory)
import json

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

# Enter the type of the object (plate, sphere, corner)
Object = 0
graph_render_mode = 0


#Parameters for incident rays 
image_width = 0
image_height = 0
Oversamp_factor = 0
Wavelength = 0.0

# variables to change the range and theta
Obj_range = 0.0
range_rcs = 0.0
degree_interval = 0.0
rx_antenna_rad = 0.0
mesh_angle_r = 0
mesh_angle_up = 0
diff_const = 0
RCS_const = 0


azimuth_angle = 0
look_at_front =  np.array([0.0, 0, 0.0])
vec_up = np.array([0.0, 0.0, 0.0])
# RCS_constant = 0.0


def calculate_coordinates(radius, degree):
   angle_rad = np.deg2rad(degree)
   # Define the named tuple format

   # Calculate x and y coordinates in meters
   From_x = np.float32(radius * np.cos(angle_rad))
   From_y = np.float32(radius * np.sin(angle_rad))
   return (From_x, From_y)

def plot_rms_and_std(error_array):
   x = np.array(range(error_array[:,0].shape[0]))
   y = error_array[:,0]
   e = error_array[:,1]
   plt.errorbar(x, y, e, linestyle='None', marker='^')

   plt.savefig("error_plot.png")

def plot_sphere_radius_performance():
   """
   method for paper plot
   """
   mpl.rc('font',family='Times New Roman')
   # gold-standard: 284sek # arround 11000 rays
   sphere_sizes = [0.5, 1.0, 2.0, 4.0]
   elapsed_sphere_times = [12.24, 4.0, 1.17, 0.43]
   elapsed_sphere_times_fast = [5.44, 1.96, 0.56, 0.28]

   elapsed_cone_times = [4.0, 2.0, 0.8, 0.71]
   elapsed_cone_times_fast = [2.23, 1.14, 0.52, 0.43]

   plt.plot(sphere_sizes, elapsed_sphere_times, marker='o', label="vary sphere size")
   plt.plot(sphere_sizes, elapsed_sphere_times_fast, marker='x', label="vary sphere size optimized")
   plt.plot(sphere_sizes, elapsed_cone_times, marker='o', label="vary cone angle")
   plt.plot(sphere_sizes, elapsed_cone_times_fast, marker='x', label="vary cone angle optimized")
   ax = plt.gca()
   ax.set_xticks(sphere_sizes)
   ax.set_xticklabels(['0.5/2°', '1.0/4.0°', '2.0/8°' ,'4.0/16°'])

   plt.legend(prop={"size":14})
   plt.xticks(fontsize=14)
   plt.yticks(fontsize=14)
   plt.grid()
   plt.xlabel("Sphere radius (m)/Cone Angle (deg.)", size=14)
   plt.ylabel("Time in s", size=14)
   plt.savefig("SphereSizePerformance.pdf", bbox_inches='tight')

def plot_antenna_configuration(radar_signal_data):

   mpl.rc('font',family='Times New Roman')
   tx_positions_2d = radar_signal_data.tx_positions[:, :]
   rx_positions_2d = radar_signal_data.rx_positions[:, :]

   tx_positions_2d -= tx_positions_2d[0]
   rx_positions_2d -= rx_positions_2d[0]
   csfont = {'fontname':'Times New Roman'}

   plt.scatter(rx_positions_2d[:, 0], rx_positions_2d[:, 1], marker='o', label="rx antenna positions")
   ax = plt.gca()
   ax.set_yticklabels([])
   ax.set_yticks([])
   ax.set_xlabel("x in m", size=14, **csfont)
   plt.xticks(**csfont)

   ax.scatter(tx_positions_2d[:, 0], tx_positions_2d[:, 1], marker='x', label="tx antenna positions")

   virtual_positions = []
   for rx_pos in rx_positions_2d:
      for tx_pos in tx_positions_2d:
         virtual_pos = (rx_pos + tx_pos)
         virtual_positions.append(virtual_pos)

   virtual_positions = np.asarray(virtual_positions)
   
   ax.scatter(virtual_positions[:, 0], virtual_positions[:, 1]-0.1, marker='o', color='blue', label="virtual antenna positions")
   ax.legend()
   ax.set_aspect(0.33)
   plt.savefig("antenna_positions.eps", bbox_inches='tight')


def get_max_range(radar_signal_data):
   c=3e8
   chirp_duration = radar_signal_data.chirp_duration
   bandwidth = radar_signal_data.bandwidth
   sample_frequency = radar_signal_data.time_vector.shape[0]/chirp_duration
   r_max = (c*chirp_duration*sample_frequency)/(4*bandwidth)*2 # complex signal multiply by 2

   return r_max

def compute_azimuth_label_sine_space(number_sample_points, angular_dim, start_sin_index=0, stop_sin_index=None):
   
   if not stop_sin_index:
      stop_sin_index = angular_dim

   start_sin_value = -2.0/angular_dim*start_sin_index + 1
   end_sin_value = -2.0/angular_dim*stop_sin_index + 1
   #sin_space_labels = np.linspace(1, -1, number_sample_points)

   sin_space_labels = np.linspace(start_sin_value, end_sin_value, number_sample_points)
   angular_labels = np.round(np.rad2deg(np.arcsin(sin_space_labels))).astype(np.int32)
   angular_positions = np.linspace(0, angular_dim-1, number_sample_points)

   #angular_labels = np.round(np.linspace(90, -90, number_sample_points))
   #angular_positions = (np.sin(np.deg2rad(angular_labels)) + 1.0)*0.5 * (angular_dim-1)
   return angular_positions, angular_labels

def compute_range_label(radar_signal_data, number_sample_points, range_dim):
   r_max = get_max_range(radar_signal_data)
   print("r_max: " + str(r_max))
   range_labels = np.round(np.linspace(0, r_max, number_sample_points)).astype(np.int32)
   range_positions = np.linspace(0, range_dim, number_sample_points)
   return range_positions, range_labels

def load_Object(renderer, material_dir, obj_filename, mesh_angle_r, mesh_angle_up, diff_const):
   mesh_list, obj_mat_list = load_mesh_normal_from_obj(obj_filename, material_dir)

   for i, obj_mat in enumerate(obj_mat_list):
      mesh = mesh_list[i]
      if "metal" in obj_mat.name.lower():
         mesh_mat = MaterialMixed(obj_mat.diffuse, diff_const)
      else:
         #mesh_mat = MaterialMetal(obj_mat.diffuse, 0.1)
         mesh_mat = MaterialMixed(obj_mat.diffuse, 0.1)
      print("Mesh loading completed")

      mesh.set_material(mesh_mat)
      mesh.rotate([1.0, 0.0, 0.0],np.deg2rad(mesh_angle_up))
      mesh.rotate([0.0, 0.0, 1.0],np.deg2rad(mesh_angle_r))
      renderer.add_geometry_object(mesh)


def calculate_azimuth_angle(Plate_size, range_rcs):
    # Calculate the half-angle theta/2 in radians
    theta_half_radians = np.arctan((Plate_size / 2) / range_rcs)
    
    # Calculate the full theta in radians
    azimuth_radians = 2 * theta_half_radians
    
    # Convert theta from radians to degrees
    azimuth_angle = np.degrees(azimuth_radians)
    
    return azimuth_angle


def load_antennas_for_imaging_iwr6843AOP(render_pos, look_at_front, vec_up,
                                         radiation_pattern, phi_axis, theta_axis, Wavelength, rx_antenna_rad, azimuth_angle):

   tx_antennas = list()
   rx_antennas = list()

   #construct the antenna array
   num_tx = 3
   num_rx = 4
   tx_antenna_pos_offset_1 = np.array([-Wavelength,0,0])
   tx_antenna_pos_offset_2 = np.array([0,-Wavelength,0])
   tx_antenna_pos_offset_3 = np.array([0,0,0])
   tx_offsets = [tx_antenna_pos_offset_1, tx_antenna_pos_offset_2, tx_antenna_pos_offset_3]
   rx_antenna_pos_offset_1 = np.array([-Wavelength,-1.5*Wavelength,0])
   rx_antenna_pos_offset_2 = np.array([-1.5*Wavelength,-1.5*Wavelength,0])
   rx_antenna_pos_offset_3 = np.array([-1.5*Wavelength,-Wavelength,0])
   rx_antenna_pos_offset_4 = np.array([-Wavelength,-Wavelength,0])
   rx_offsets = [rx_antenna_pos_offset_1, rx_antenna_pos_offset_2, rx_antenna_pos_offset_3, rx_antenna_pos_offset_4]

   for i in range(num_tx):
      tx_antenna_pos = render_pos + tx_offsets[i] #for i= 0 [0.2951,0,0]
      #tx_antenna_pos = tx_antennas_pos[i] + camera_pos_front
      tx_antenna = TxAntenna(tx_antenna_pos)
      tx_antenna.set_look_at(look_at_front)
      tx_antenna.set_up(vec_up)
      tx_antenna.set_azimuth(np.deg2rad(azimuth_angle))
      tx_antenna.set_elevation(np.deg2rad(azimuth_angle))
      tx_antenna.set_radiation_pattern(radiation_pattern, phi_axis, theta_axis)
      tx_antennas.append(tx_antenna)
         
   # add rx antenna pos 
   for i in range(num_rx):
      rx_antenna_pos = render_pos + rx_offsets[i]
      rx_antenna = RxAntenna(rx_antenna_pos, rx_antenna_rad)
      rx_antenna.set_look_at(look_at_front)
      rx_antenna.set_up(vec_up)
      rx_antenna.set_radiation_pattern(radiation_pattern, phi_axis, theta_axis)
      rx_antennas.append(rx_antenna)

   return tx_antennas, rx_antennas

def simulate_plate(image_width, image_height, Oversamp_factor, Wavelength, 
                      rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle_rad,
                      mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type, diff_const, RCS_const):
   renderer = Renderer()
   render_mode = getattr(RenderMode, render_mode_type)
   
   # load mesh from outside

   script_directory = os.path.dirname(__file__)
   content_directory = os.path.join(script_directory, "../example-files/Avinash_RCS/Plate/")
   obj_filename = os.path.join(content_directory, "Plate_high_roughness_15cm.obj")#Plate_high_roughness_metal, Plate_high_roughness_flipped
   #print("start loading campus scene")
   load_Object(renderer, content_directory, obj_filename, mesh_angle_r, mesh_angle_up, diff_const)

   radiation_pattern_filename = os.path.join(script_directory, "../example-files/Ant_Pattern_onChip_310GHz_Einzelelement_v2-unmodifiedCopy.txt")
   #radiation_pattern_filename = os.path.join(script_directory, "../example-files/test_radiation_pattern.txt")
   radiation_pattern, phi_axis, theta_axis = load_radiation_pattern_from_cst(radiation_pattern_filename)
   #plot_radiation_pattern_3d(radiation_pattern, phi_axis, theta_axis, subsampling_factor=8)
   #plt.show()


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
      print("Image stored in RCS_Plate_5cm")
      image_array.save("RCS_Plate_5cm.png")

   elif render_mode == RenderMode.RENDER_MODE_RAYTARGET_COMP:
      # This mode supports accurate Doppler and Radiation-Patterns
      
      tx_antennas, rx_antennas = load_antennas_for_imaging_iwr6843AOP(From_vector, look_at_front, vec_up,
                                         radiation_pattern, phi_axis, theta_axis, Wavelength, rx_antenna_rad, azimuth_angle_rad)
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
      print(f" RCS in linear {RCS_with_const}")
      RCS_with_const_dBsm = 10 * np.log10(RCS_with_const) 
      print(f" RCS in dBsm {RCS_with_const_dBsm}") 

      return RCS_with_const_dBsm
      # Converting gains from dB to linear scale
      # RCS_final = (RCS_without_const )
      # print(f" RCS in dBsm {RCS_final}")
      # print(f" RCS with const factor linear {RCS_final}") 
      # RCS_dBsm = 10 * np.log10(RCS_final) 
      # print(f" RCS in dBsm {RCS_dBsm}")

def simulate_corner(image_width, image_height, Oversamp_factor, Wavelength, 
                      rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle,
                      mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type, diff_const, RCS_const):

   renderer = Renderer()
   render_mode = getattr(RenderMode, render_mode_type)
   
   # load mesh from outside

   script_directory = os.path.dirname(__file__)
   content_directory = os.path.join(script_directory, "../example-files/Avinash_RCS/Corner/")#
   obj_filename = os.path.join(content_directory, "Corner_15cm.obj")#Corner_MOM24_mesh,Corner_MOM24_mesh, Corner_MOM24_withmetal , Corner_7cm
   #print("start loading campus scene")
   load_Object(renderer, content_directory, obj_filename, mesh_angle_r, mesh_angle_up, diff_const)

   radiation_pattern_filename = os.path.join(script_directory, "../example-files/Ant_Pattern_onChip_310GHz_Einzelelement_v2-unmodifiedCopy.txt")
   #radiation_pattern_filename = os.path.join(script_directory, "../example-files/test_radiation_pattern.txt")
   radiation_pattern, phi_axis, theta_axis = load_radiation_pattern_from_cst(radiation_pattern_filename)
   #plot_radiation_pattern_3d(radiation_pattern, phi_axis, theta_axis, subsampling_factor=8)
   #plt.show()

   
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
      print("Image stored in RCS_corner_rotate")
      image_array.save("RCS_corner_rotate.png")
   elif render_mode == RenderMode.RENDER_MODE_RAYTARGET_COMP:
      # This mode supports accurate Doppler and Radiation-Patterns
      
      tx_antennas, rx_antennas = load_antennas_for_imaging_iwr6843AOP(From_vector, look_at_front, vec_up,
                                         radiation_pattern, phi_axis, theta_axis, Wavelength, rx_antenna_rad, azimuth_angle)
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
      print(f" power ratio {Power_ratio}")
      RCS_without_const = Power_ratio*(4*np.pi*((Obj_range)**2))**2
      RCS_without_const_dBsm = 10 * np.log10(RCS_without_const) 
      print(f" RCS in dBsm without const factor{RCS_without_const_dBsm}")
      RCS_with_const = RCS_without_const*RCS_const
      # Converting gains from dB to linear scale 
      RCS_with_const_dBsm = 10 * np.log10(RCS_with_const) 
      print(f" RCS in dBsm with const factor{RCS_with_const_dBsm}")
      return RCS_with_const_dBsm

def simulate_sphere(image_width, image_height, Oversamp_factor, Wavelength, 
                     rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle_rad,
                     mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type, diff_const, RCS_const):

   renderer = Renderer()
   render_mode = getattr(RenderMode, render_mode_type)
   print(f"The diff_const is {diff_const}")
   # load mesh from outside

   script_directory = os.path.dirname(__file__)
   content_directory = os.path.join(script_directory, "../example-files/Avinash_RCS/Sphere/")
   obj_filename = os.path.join(content_directory, "sphere_15cm.obj")#Avinash_sphere, Spehere_RCS_metal
   #print("start loading campus scene")
   load_Object(renderer, content_directory, obj_filename, mesh_angle_r, mesh_angle_up, diff_const)

   radiation_pattern_filename = os.path.join(script_directory, "../example-files/Ant_Pattern_onChip_310GHz_Einzelelement_v2-unmodifiedCopy.txt")
   #radiation_pattern_filename = os.path.join(script_directory, "../example-files/test_radiation_pattern.txt")
   radiation_pattern, phi_axis, theta_axis = load_radiation_pattern_from_cst(radiation_pattern_filename)
   # plot_radiation_pattern_3d(radiation_pattern, phi_axis, theta_axis)
   # plt.show()

   # set tx and rx antennas
   # antenna_offset = np.array([-8.8, 32.0, 0.7])

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
      print("Image stored in RCS_sphere_5cm")
      image_array.save("RCS_sphere_5cm.png")
   elif render_mode == RenderMode.RENDER_MODE_RAYTARGET_COMP:
      # This mode supports accurate Doppler and Radiation-Patterns
      
      tx_antennas, rx_antennas = load_antennas_for_imaging_iwr6843AOP(From_vector, look_at_front, vec_up, radiation_pattern, phi_axis, theta_axis, Wavelength, rx_antenna_rad, azimuth_angle_rad)
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
      print(f" RCS without const factor linear {RCS_without_const}")
      RCS_with_const = RCS_without_const * RCS_const
      RCS_with_const_dBsm = 10 * np.log10(RCS_with_const) 
      print(f" RCS with const factor dBsm {RCS_with_const_dBsm}") 
      return RCS_with_const_dBsm
      # Converting gains from dB to linear scale
      # RCS_final = (RCS_without_const)
      # print(f" RCS with const linear {RCS_final}")
      # # print(f" RCS with const factor linear {RCS_final}") 
      # RCS_dBsm = 10 * np.log10(RCS_final) 

      # print(f" RCS with const factor dBsm {RCS_dBsm}")


# def save_to_json(Obj_range, mesh_angle_r, mesh_angle_up, RCS_without_const_dBsm, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle):
#     # Initialize data structure with 'Measurements' key
#     data = {'Measurements': []}

#     # Load existing data if the file exists and is not empty
#     if os.path.exists(json_filename) and os.path.getsize(json_filename) > 0:
#         with open(json_filename, 'r') as file:
#             try:
#                 existing_data = json.load(file)
#                 if 'Measurements' in existing_data:
#                     data['Measurements'] = existing_data['Measurements']
#                 else:
#                     print("No 'Measurements' key in JSON file. Initializing with empty list.")
#             except json.JSONDecodeError:
#                 print(f"Error decoding JSON from {json_filename}. Starting with an empty dataset.")

#     # Check if this specific measurement already exists
#     existing_entry = None
#     for entry in data['Measurements']:
#         if (entry['Obj_range'] == Obj_range and entry['RCS_without_const_dBsm'] == RCS_without_const_dBsm):
#             existing_entry = entry
#             break

#     if existing_entry:
#         # Update existing entry
#         existing_entry['Oversamp_factor'] = Oversamp_factor
#         existing_entry['rx_antenna_rad'] = rx_antenna_rad
#         existing_entry['azimuth_angle'] = azimuth_angle
#         existing_entry['mesh_angle_r'] = mesh_angle_r
#         existing_entry['mesh_angle_up'] = mesh_angle_up
#     else:
#         # Create and append a new entry if not found
#         new_entry = {
#             'Obj_range': Obj_range,
#             'RCS_without_const_dBsm': RCS_without_const_dBsm,
#             'Oversamp_factor': Oversamp_factor,
#             'rx_antenna_rad': rx_antenna_rad,
#             'azimuth_angle': azimuth_angle,
#             'mesh_angle_r':  mesh_angle_r,
#             'mesh_angle_up':  mesh_angle_up
#         }
#         data['Measurements'].append(new_entry)

#     # Write the updated or new data back to the file
#     with open(json_filename, 'w') as file:
#         json.dump(data, file, indent=4)
#     print(f"Data saved to respected file")

def save_to_json(Obj_range, mesh_angle_r, mesh_angle_up, RCS_with_const_dBsm, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle, diff_const):
    # Initialize data structure with 'Measurements' key
    data = {'Measurements': []}

    # Attempt to open the file and read existing data
    if os.path.exists(json_filename) and os.path.getsize(json_filename) > 0:
      try:
         with open(json_filename, 'r') as file:
               existing_data = json.load(file)
               if 'Measurements' in existing_data:
                  data['Measurements'] = existing_data['Measurements']
               else:
                  print("No 'Measurements' key in JSON file. Initializing with empty list.")
      except (FileNotFoundError, json.JSONDecodeError):
         print(f"Error reading {json_filename}. Starting with an empty dataset.")

    # Create a new entry
    new_entry = {
        'Obj_range': Obj_range,
        'RCS_with_const_dBsm': RCS_with_const_dBsm,
        'Oversamp_factor': Oversamp_factor,
        'rx_antenna_rad': rx_antenna_rad,
        'azimuth_angle': azimuth_angle,
        'mesh_angle_r': mesh_angle_r,
        'mesh_angle_up': mesh_angle_up,
        'diff_const':diff_const 
    }
    data['Measurements'].append(new_entry)

    # Write the updated or new data back to the file
    with open(json_filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to {json_filename}")

def main():
   # normal mode
   # 0.1m
   # original from paper
   #comparison_image = simulate((800, 600), oversampling_factor=18, rx_radius=0.10, \
   #   number_sequences=120, enable_color_bar=True, enable_range_axis=False, filename="fft_image_campus_10cm_2.png")

   '''
   '''


   #Parameters for incident rays 
   image_width = 1200
   image_height = 600
   Oversamp_factor = 30
   Wavelength = 0.005

   # variables to change the range and theta
   Obj_range = 2
   iterations = 1
   range_azimuth = 2
   rx_antenna_rad = 0.0


   look_at_front =  np.array([0.0, 0, 0.0])
   vec_up = np.array([0.0, 0.0, 1.0]) 
   
   RCS_const = 0
   # Enter the type of the object (plate, sphere, corner)
   Object = "corner".lower()

   # adjusted for smaller gpus
   if Object == "sphere":
      render_mode_type = "RENDER_MODE_RAYTARGET_COMP"   #RENDER_MODE_RAYTARGET_COMP,  RENDER_MODE_GRAPHICS
      mesh_angle_r = 0
      mesh_angle_up = 0
      sphere_dimention = 0.1
      diff_const = 0.0
      RCS_const = 188.1575 #1.8440
      # From_vector = np.array([Obj_range, 0.0, 0.0])
      azimuth_angle = calculate_azimuth_angle(sphere_dimention, range_azimuth)
      print(f"The azimuth angle is set to {azimuth_angle:.2f} degrees.")
      # azimuth_angle_rad = np.deg2rad(azimuth_angle)
      # simulate_sphere(image_width, image_height, Oversamp_factor, Wavelength, 
      #                 rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle,
      #                 mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type)
      
      script_directory = os.path.dirname(__file__)
      jsonfile_directory = os.path.join(script_directory, "../example-scripts/Avinash_json_files/Sphere_Json")
      json_filename = os.path.join(jsonfile_directory, "RCS_sphere_dist_results_auto.json")
      for i in range(iterations):
         Obj_range += 0
         From_vector = np.array([Obj_range, 0.0, 0.0])
         print(f"The Obj_range is {Obj_range:.2f}")
         RCS_without_const_dBsm = simulate_sphere(image_width, image_height, Oversamp_factor, Wavelength, 
                      rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle,
                      mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type, diff_const, RCS_const)
         time.sleep(4)
         # save_to_json(Obj_range, mesh_angle_r, mesh_angle_up, RCS_without_const_dBsm, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle)

   
   elif Object == "plate":
      render_mode_type = "RENDER_MODE_RAYTARGET_COMP"   #RENDER_MODE_RAYTARGET_COMP,  RENDER_MODE_GRAPHICS
      plate_dimention = 0.05
      mesh_angle_r = 0
      mesh_angle_up = 0
      diff_const = 0.0
      rx_antenna_rad = 0.3
      RCS_const = 0.3164 
      
      azimuth_angle =  calculate_azimuth_angle(plate_dimention, range_azimuth)
      
      # azimuth_angle_rad = np.deg2rad(azimuth_angle)
      # RCS_without_const_dBsm = simulate_plate(image_width, image_height, Oversamp_factor, Wavelength, 
      #                 rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle,
      #                 mesh_angle_r, Obj_range, render_mode_type)
      # azimuth_angle_rad = np.deg2rad(azimuth_angle)

      script_directory = os.path.dirname(__file__)
      jsonfile_directory = os.path.join(script_directory, "../example-scripts/Avinash_json_files/Plate_Json/")
      json_filename = os.path.join(jsonfile_directory, "RCS_plate_vs_azimuth_elevation.json")
      for i in range(iterations):
         azimuth_angle += 1
         From_vector = np.array([Obj_range, 0.0, 0.0])
         # print(f"The Obj_range is {Obj_range:.2f}")
         print(f"The Obj_range is {Obj_range:.2f}")
         print(f"The azimuth angle should be set to {azimuth_angle:.2f} degrees.")  
         RCS_with_const_dBsm = simulate_plate(image_width, image_height, Oversamp_factor, Wavelength, 
                      rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle,
                      mesh_angle_r, mesh_angle_up,  Obj_range, render_mode_type, diff_const, RCS_const)
         time.sleep(3)
         save_to_json(Obj_range, mesh_angle_r, mesh_angle_up, RCS_with_const_dBsm, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle, diff_const)

   
   elif Object == "corner":
      render_mode_type = "RENDER_MODE_RAYTARGET_COMP"   #RENDER_MODE_RAYTARGET_COMP,  RENDER_MODE_GRAPHICS
      corner_dimention = 0.08
      mesh_angle_r = 0
      mesh_angle_up = 45
      rx_antenna_rad = 0.3
      diff_const = 0.0
      RCS_const = 0.42465
      # From_vector = np.array([0.0, -(Obj_range), 0.0])
      script_directory = os.path.dirname(__file__)
      jsonfile_directory = os.path.join(script_directory, "../example-scripts/Avinash_json_files/Corner_Json/")
      json_filename = os.path.join(jsonfile_directory, "RCS_corner_vs_azimuth_elevation.json")
      azimuth_angle = calculate_azimuth_angle(corner_dimention, range_azimuth)
      
      #azimuth_angle_rad = np.deg2rad(azimuth_angle)
      # simulate_corner(image_width, image_height, Oversamp_factor, Wavelength, 
      #                 rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle,
      #                 mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type)
      
      for i in range(iterations):
         azimuth_angle += 1
         print(f"The azimuth angle should be set to {azimuth_angle:.2f} degrees.")
         # From_vector = np.array([0.0, -(Obj_range), 0.0])
         print(f"The rx_antenna_rad is {rx_antenna_rad:.2f}")
         From_vector = np.array([0.0, -(Obj_range), 0.0])
         RCS_without_const_dBsm = simulate_corner(image_width, image_height, Oversamp_factor, Wavelength, 
                       rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle,
                       mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type, diff_const, RCS_const)
         time.sleep(3)
         save_to_json(Obj_range, mesh_angle_r, mesh_angle_up, RCS_without_const_dBsm, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle, diff_const)

   else:
      print("Invalid Object type.")
      return
   
   # script_directory = os.path.dirname(__file__)
   # jsonfile_directory = os.path.join(script_directory, "../example-scripts/")
   # json_filename = os.path.join(jsonfile_directory, "RCS_plate_results_auto.json")
   # save_to_json(Obj_range, RCS_without_const_dBsm, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle)

if __name__ == "__main__":
   main()
   print("Finished")




























































7/6/2024



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

# Enter the type of the object (plate, sphere, corner)
Object = 0
graph_render_mode = 0


#Parameters for incident rays 
image_width = 0
image_height = 0
Oversamp_factor = 0
Wavelength = 0.0

# variables to change the range and theta
Obj_range = 0.0
range_rcs = 0.0
degree_interval = 0.0
rx_antenna_rad = 0.0
mesh_angle_r = 0
mesh_angle_up = 0
diff_const = 0
RCS_const = 0
RCS_with_const = 0


azimuth_angle = 0
look_at_front =  np.array([0.0, 0, 0.0])
vec_up = np.array([0.0, 0.0, 0.0])
# RCS_constant = 0.0


def calculate_coordinates(radius, degree):
   angle_rad = np.deg2rad(degree)
   # Define the named tuple format

   # Calculate x and y coordinates in meters
   From_x = np.float32(radius * np.cos(angle_rad))
   From_y = np.float32(radius * np.sin(angle_rad))
   return (From_x, From_y)

def plot_rms_and_std(error_array):
   x = np.array(range(error_array[:,0].shape[0]))
   y = error_array[:,0]
   e = error_array[:,1]
   plt.errorbar(x, y, e, linestyle='None', marker='^')

   plt.savefig("error_plot.png")

def plot_sphere_radius_performance():
   """
   method for paper plot
   """
   mpl.rc('font',family='Times New Roman')
   # gold-standard: 284sek # arround 11000 rays
   sphere_sizes = [0.5, 1.0, 2.0, 4.0]
   elapsed_sphere_times = [12.24, 4.0, 1.17, 0.43]
   elapsed_sphere_times_fast = [5.44, 1.96, 0.56, 0.28]

   elapsed_cone_times = [4.0, 2.0, 0.8, 0.71]
   elapsed_cone_times_fast = [2.23, 1.14, 0.52, 0.43]

   plt.plot(sphere_sizes, elapsed_sphere_times, marker='o', label="vary sphere size")
   plt.plot(sphere_sizes, elapsed_sphere_times_fast, marker='x', label="vary sphere size optimized")
   plt.plot(sphere_sizes, elapsed_cone_times, marker='o', label="vary cone angle")
   plt.plot(sphere_sizes, elapsed_cone_times_fast, marker='x', label="vary cone angle optimized")
   ax = plt.gca()
   ax.set_xticks(sphere_sizes)
   ax.set_xticklabels(['0.5/2°', '1.0/4.0°', '2.0/8°' ,'4.0/16°'])

   plt.legend(prop={"size":14})
   plt.xticks(fontsize=14)
   plt.yticks(fontsize=14)
   plt.grid()
   plt.xlabel("Sphere radius (m)/Cone Angle (deg.)", size=14)
   plt.ylabel("Time in s", size=14)
   plt.savefig("SphereSizePerformance.pdf", bbox_inches='tight')

def plot_antenna_configuration(radar_signal_data):

   mpl.rc('font',family='Times New Roman')
   tx_positions_2d = radar_signal_data.tx_positions[:, :]
   rx_positions_2d = radar_signal_data.rx_positions[:, :]

   tx_positions_2d -= tx_positions_2d[0]
   rx_positions_2d -= rx_positions_2d[0]
   csfont = {'fontname':'Times New Roman'}

   plt.scatter(rx_positions_2d[:, 0], rx_positions_2d[:, 1], marker='o', label="rx antenna positions")
   ax = plt.gca()
   ax.set_yticklabels([])
   ax.set_yticks([])
   ax.set_xlabel("x in m", size=14, **csfont)
   plt.xticks(**csfont)

   ax.scatter(tx_positions_2d[:, 0], tx_positions_2d[:, 1], marker='x', label="tx antenna positions")

   virtual_positions = []
   for rx_pos in rx_positions_2d:
      for tx_pos in tx_positions_2d:
         virtual_pos = (rx_pos + tx_pos)
         virtual_positions.append(virtual_pos)

   virtual_positions = np.asarray(virtual_positions)
   
   ax.scatter(virtual_positions[:, 0], virtual_positions[:, 1]-0.1, marker='o', color='blue', label="virtual antenna positions")
   ax.legend()
   ax.set_aspect(0.33)
   plt.savefig("antenna_positions.eps", bbox_inches='tight')


def get_max_range(radar_signal_data):
   c=3e8
   chirp_duration = radar_signal_data.chirp_duration
   bandwidth = radar_signal_data.bandwidth
   sample_frequency = radar_signal_data.time_vector.shape[0]/chirp_duration
   r_max = (c*chirp_duration*sample_frequency)/(4*bandwidth)*2 # complex signal multiply by 2

   return r_max

def compute_azimuth_label_sine_space(number_sample_points, angular_dim, start_sin_index=0, stop_sin_index=None):
   
   if not stop_sin_index:
      stop_sin_index = angular_dim

   start_sin_value = -2.0/angular_dim*start_sin_index + 1
   end_sin_value = -2.0/angular_dim*stop_sin_index + 1
   #sin_space_labels = np.linspace(1, -1, number_sample_points)

   sin_space_labels = np.linspace(start_sin_value, end_sin_value, number_sample_points)
   angular_labels = np.round(np.rad2deg(np.arcsin(sin_space_labels))).astype(np.int32)
   angular_positions = np.linspace(0, angular_dim-1, number_sample_points)

   #angular_labels = np.round(np.linspace(90, -90, number_sample_points))
   #angular_positions = (np.sin(np.deg2rad(angular_labels)) + 1.0)*0.5 * (angular_dim-1)
   return angular_positions, angular_labels

def compute_range_label(radar_signal_data, number_sample_points, range_dim):
   r_max = get_max_range(radar_signal_data)
   print("r_max: " + str(r_max))
   range_labels = np.round(np.linspace(0, r_max, number_sample_points)).astype(np.int32)
   range_positions = np.linspace(0, range_dim, number_sample_points)
   return range_positions, range_labels

def load_Object(renderer, material_dir, obj_filename, mesh_angle_r, mesh_angle_up, diff_const):
   mesh_list, obj_mat_list = load_mesh_normal_from_obj(obj_filename, material_dir)

   for i, obj_mat in enumerate(obj_mat_list):
      mesh = mesh_list[i]
      if "metal" in obj_mat.name.lower():
         mesh_mat = MaterialMixed(obj_mat.diffuse, diff_const)
      else:
         #mesh_mat = MaterialMetal(obj_mat.diffuse, 0.1)
         mesh_mat = MaterialMixed(obj_mat.diffuse, 0.1)
      print("Mesh loading completed")

      mesh.set_material(mesh_mat)
      mesh.rotate([1.0, 0.0, 0.0],np.deg2rad(mesh_angle_up))
      mesh.rotate([0.0, 0.0, 1.0],np.deg2rad(mesh_angle_r))
      renderer.add_geometry_object(mesh)


def calculate_azimuth_angle(obj_width, range_rcs):
    # Calculate the half-angle theta/2 in radians
    theta_half_radians = np.arctan((obj_width / 2) / range_rcs)
    
    # Calculate the full theta in radians
    azimuth_radians = 2 * theta_half_radians
    
    # Convert theta from radians to degrees
    azimuth_angle = np.degrees(azimuth_radians)
    
    return azimuth_angle

def calculate_elevation_angle(obj_length, range_rcs):
    # Calculate the half-angle theta/2 in radians
    theta_half_radians = np.arctan((obj_length / 2) / range_rcs)
    
    # Calculate the full theta in radians
    elevation_radians = 2 * theta_half_radians
    
    # Convert theta from radians to degrees
    elevation_angle = np.degrees(elevation_radians)
    
    return elevation_angle

      # Define the integrand function
def integrand(theta, phi):
   # Use the single RCS value directly in linear scale
   rcs_value = RCS_with_const 
   # Return the product of RCS and cosine of the elevation angle
   # print(f"rcs_value {rcs_value} rcs_value inside the integrand")
   return rcs_value * np.cos(theta)


def load_antennas_for_imaging_iwr6843AOP(render_pos, look_at_front, vec_up,
                                         radiation_pattern, phi_axis, theta_axis, Wavelength, rx_antenna_rad, azimuth_angle, elevation_angle):

   tx_antennas = list()
   rx_antennas = list()

   #construct the antenna array
   num_tx = 3
   num_rx = 4
   tx_antenna_pos_offset_1 = np.array([-Wavelength,0,0])
   tx_antenna_pos_offset_2 = np.array([0,-Wavelength,0])
   tx_antenna_pos_offset_3 = np.array([0,0,0])
   tx_offsets = [tx_antenna_pos_offset_1, tx_antenna_pos_offset_2, tx_antenna_pos_offset_3]
   rx_antenna_pos_offset_1 = np.array([-Wavelength,-1.5*Wavelength,0])
   rx_antenna_pos_offset_2 = np.array([-1.5*Wavelength,-1.5*Wavelength,0])
   rx_antenna_pos_offset_3 = np.array([-1.5*Wavelength,-Wavelength,0])
   rx_antenna_pos_offset_4 = np.array([-Wavelength,-Wavelength,0])
   rx_offsets = [rx_antenna_pos_offset_1, rx_antenna_pos_offset_2, rx_antenna_pos_offset_3, rx_antenna_pos_offset_4]

   for i in range(num_tx):
      tx_antenna_pos = render_pos + tx_offsets[i] #for i= 0 [0.2951,0,0]
      #tx_antenna_pos = tx_antennas_pos[i] + camera_pos_front
      tx_antenna = TxAntenna(tx_antenna_pos)
      tx_antenna.set_look_at(look_at_front)
      tx_antenna.set_up(vec_up)
      tx_antenna.set_azimuth(np.deg2rad(azimuth_angle))
      tx_antenna.set_elevation(np.deg2rad(elevation_angle))
      tx_antenna.set_radiation_pattern(radiation_pattern, phi_axis, theta_axis)
      tx_antennas.append(tx_antenna)
         
   # add rx antenna pos 
   for i in range(num_rx):
      rx_antenna_pos = render_pos + rx_offsets[i]
      rx_antenna = RxAntenna(rx_antenna_pos, rx_antenna_rad)
      rx_antenna.set_look_at(look_at_front)
      rx_antenna.set_up(vec_up)
      rx_antenna.set_radiation_pattern(radiation_pattern, phi_axis, theta_axis)
      rx_antennas.append(rx_antenna)

   return tx_antennas, rx_antennas

def simulate_plate(image_width, image_height, Oversamp_factor, Wavelength, 
                      rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle, elevation_angle,
                      mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type, diff_const, RCS_const):
   renderer = Renderer()
   render_mode = getattr(RenderMode, render_mode_type)
   
   # load mesh from outside

   script_directory = os.path.dirname(__file__)
   content_directory = os.path.join(script_directory, "../example-files/Avinash_RCS/Plate/")
   obj_filename = os.path.join(content_directory, "Plate_high_roughness_05cm.obj")#Plate_high_roughness_metal, Plate_high_roughness_flipped
   #print("start loading campus scene")
   load_Object(renderer, content_directory, obj_filename, mesh_angle_r, mesh_angle_up, diff_const)

   radiation_pattern_filename = os.path.join(script_directory, "../example-files/Ant_Pattern_onChip_310GHz_Einzelelement_v2-unmodifiedCopy.txt")
   #radiation_pattern_filename = os.path.join(script_directory, "../example-files/test_radiation_pattern.txt")
   radiation_pattern, phi_axis, theta_axis = load_radiation_pattern_from_cst(radiation_pattern_filename)
   #plot_radiation_pattern_3d(radiation_pattern, phi_axis, theta_axis, subsampling_factor=8)
   #plt.show()


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
      print("Image stored in RCS_Plate_checking_human")
      image_array.save("RCS_Plate_checking_human.png")

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
      print(f"RCS_with_const {RCS_with_const}")

      azimuth_radi = np.deg2rad(azimuth_angle)
      elevation_radi = np.deg2rad(elevation_angle)

      # For the effective area, performing double integral over azimuth and elevation, which provides effective area in the omni directional sphere 
      effective_area, error = dblquad(lambda phi, theta: np.cos(theta),
                                -azimuth_radi / 2, azimuth_radi / 2,  # Azimuth angle range
                                lambda x: -elevation_radi / 2, lambda x: elevation_radi / 2)  # Elevation angle range

      # To calculate the gain factor
      gain_factor = (4 * np.pi) / effective_area

      print(f"gain_factor : {gain_factor:.4f}")

      # RCS_with_const_gain = RCS_with_const / gain_factor

      RCS_with_const_dBsm = 10 * np.log10(RCS_with_const) 

      print(f" RCS in dBsm {RCS_with_const_dBsm}") 

      

      return RCS_with_const_dBsm, gain_factor



def simulate_corner(image_width, image_height, Oversamp_factor, Wavelength, 
                      rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle, elevation_angle,
                      mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type, diff_const, RCS_const):

   renderer = Renderer()
   render_mode = getattr(RenderMode, render_mode_type)
   
   # load mesh from outside

   script_directory = os.path.dirname(__file__)
   content_directory = os.path.join(script_directory, "../example-files/Avinash_RCS/Corner/")#
   obj_filename = os.path.join(content_directory, "Corner_15cm.obj")#Corner_MOM24_mesh,Corner_MOM24_mesh, Corner_MOM24_withmetal , Corner_7cm
   #print("start loading campus scene")
   load_Object(renderer, content_directory, obj_filename, mesh_angle_r, mesh_angle_up, diff_const)

   radiation_pattern_filename = os.path.join(script_directory, "../example-files/Ant_Pattern_onChip_310GHz_Einzelelement_v2-unmodifiedCopy.txt")
   #radiation_pattern_filename = os.path.join(script_directory, "../example-files/test_radiation_pattern.txt")
   radiation_pattern, phi_axis, theta_axis = load_radiation_pattern_from_cst(radiation_pattern_filename)
   #plot_radiation_pattern_3d(radiation_pattern, phi_axis, theta_axis, subsampling_factor=8)
   #plt.show()

   
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
      print("Image stored in RCS_corner_rotate")
      image_array.save("RCS_corner_rotate.png")
      
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


      RCS_without_const = Power_ratio*(4*np.pi*((Obj_range)**2))**2
      
      RCS_with_const = RCS_without_const*RCS_const

      print(f"RCS_with_const {RCS_with_const}")

      azimuth_radi = np.deg2rad(azimuth_angle)
      elevation_radi = np.deg2rad(elevation_angle)

      #  Performing the double integral over the azimuth and elevation
      effective_area, error = dblquad(lambda phi,theta:  np.cos(theta),
                                -azimuth_radi / 2, azimuth_radi / 2,  # Azimuth angle range
                                lambda x: -elevation_radi / 2, lambda x: elevation_radi / 2)  # Elevation angle range

      # Calculate the gain factor
      gain_factor = (4 * np.pi) / effective_area

      print(f"gain_factor : {gain_factor:.4f}")

      RCS_with_const_dBsm = 10 * np.log10(RCS_with_const)

      print(f" RCS in dBsm {RCS_with_const_dBsm}") 

      return RCS_with_const_dBsm, gain_factor


def simulate_sphere(image_width, image_height, Oversamp_factor, Wavelength, 
                     rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle_rad,elevation_angle,
                     mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type, diff_const, RCS_const):

   renderer = Renderer()
   render_mode = getattr(RenderMode, render_mode_type)
   print(f"The diff_const is {diff_const}")
   # load mesh from outside

   script_directory = os.path.dirname(__file__)
   content_directory = os.path.join(script_directory, "../example-files/Avinash_RCS/Sphere/")
   obj_filename = os.path.join(content_directory, "sphere_15cm.obj")#Avinash_sphere, Spehere_RCS_metal
   #print("start loading campus scene")
   load_Object(renderer, content_directory, obj_filename, mesh_angle_r, mesh_angle_up, diff_const)

   radiation_pattern_filename = os.path.join(script_directory, "../example-files/Ant_Pattern_onChip_310GHz_Einzelelement_v2-unmodifiedCopy.txt")
   #radiation_pattern_filename = os.path.join(script_directory, "../example-files/test_radiation_pattern.txt")
   radiation_pattern, phi_axis, theta_axis = load_radiation_pattern_from_cst(radiation_pattern_filename)
   plot_radiation_pattern_3d(radiation_pattern, phi_axis, theta_axis)
   plt.show()

   # set tx and rx antennas
   # antenna_offset = np.array([-8.8, 32.0, 0.7])

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
      print("Image stored in RCS_sphere_5cm")
      image_array.save("RCS_sphere_5cm.png")
   elif render_mode == RenderMode.RENDER_MODE_RAYTARGET_COMP:
      # This mode supports accurate Doppler and Radiation-Patterns
      
      tx_antennas, rx_antennas = load_antennas_for_imaging_iwr6843AOP(From_vector, look_at_front, vec_up, radiation_pattern, phi_axis, theta_axis, Wavelength, rx_antenna_rad, azimuth_angle_rad, elevation_angle)
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
      print(f" RCS without const factor linear {RCS_without_const}")
      RCS_with_const = RCS_without_const * RCS_const
      RCS_with_const_dBsm = 10 * np.log10(RCS_with_const) 
      print(f" RCS with const factor dBsm {RCS_with_const_dBsm}") 
      return RCS_with_const_dBsm
      # Converting gains from dB to linear scale
      # RCS_final = (RCS_without_const)
      # print(f" RCS with const linear {RCS_final}")
      # # print(f" RCS with const factor linear {RCS_final}") 
      # RCS_dBsm = 10 * np.log10(RCS_final) 

      # print(f" RCS with const factor dBsm {RCS_dBsm}")


def simulate_human(image_width, image_height, Oversamp_factor, Wavelength, 
                     rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle_rad,elevation_angle,
                     mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type, diff_const, RCS_const):

   renderer = Renderer()
   render_mode = getattr(RenderMode, render_mode_type)
   print(f"The diff_const is {diff_const}")
   # load mesh from outside

   script_directory = os.path.dirname(__file__)
   content_directory = os.path.join(script_directory, "../example-files/Avinash_RCS/Human/")
   obj_filename = os.path.join(content_directory, "Peter_RCS.obj")#Avinash_sphere, Spehere_RCS_metal
   #print("start loading campus scene")
   load_Object(renderer, content_directory, obj_filename, mesh_angle_r, mesh_angle_up, diff_const)

   radiation_pattern_filename = os.path.join(script_directory, "../example-files/Ant_Pattern_onChip_310GHz_Einzelelement_v2-unmodifiedCopy.txt")
   #radiation_pattern_filename = os.path.join(script_directory, "../example-files/test_radiation_pattern.txt")
   radiation_pattern, phi_axis, theta_axis = load_radiation_pattern_from_cst(radiation_pattern_filename)
   # plot_radiation_pattern_3d(radiation_pattern, phi_axis, theta_axis)
   # plt.show()

   # set tx and rx antennas
   # antenna_offset = np.array([-8.8, 32.0, 0.7])

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
      print("Image stored in RCS_human")
      image_array.save("RCS_human.png")
   elif render_mode == RenderMode.RENDER_MODE_RAYTARGET_COMP:
      # This mode supports accurate Doppler and Radiation-Patterns
      
      tx_antennas, rx_antennas = load_antennas_for_imaging_iwr6843AOP(From_vector, look_at_front, vec_up, radiation_pattern, phi_axis, theta_axis, Wavelength, rx_antenna_rad, azimuth_angle_rad, elevation_angle)
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
      print(f" RCS without const factor linear {RCS_without_const}")
      RCS_with_const = RCS_without_const * RCS_const
      RCS_with_const_dBsm = 10 * np.log10(RCS_with_const) 
      print(f" RCS with const factor dBsm {RCS_with_const_dBsm}") 
      gain_factor = 1
      return RCS_with_const_dBsm, gain_factor

# def save_to_json(Obj_range, mesh_angle_r, mesh_angle_up, RCS_without_const_dBsm, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle):
#     # Initialize data structure with 'Measurements' key
#     data = {'Measurements': []}

#     # Load existing data if the file exists and is not empty
#     if os.path.exists(json_filename) and os.path.getsize(json_filename) > 0:
#         with open(json_filename, 'r') as file:
#             try:
#                 existing_data = json.load(file)
#                 if 'Measurements' in existing_data:
#                     data['Measurements'] = existing_data['Measurements']
#                 else:
#                     print("No 'Measurements' key in JSON file. Initializing with empty list.")
#             except json.JSONDecodeError:
#                 print(f"Error decoding JSON from {json_filename}. Starting with an empty dataset.")

#     # Check if this specific measurement already exists
#     existing_entry = None
#     for entry in data['Measurements']:
#         if (entry['Obj_range'] == Obj_range and entry['RCS_without_const_dBsm'] == RCS_without_const_dBsm):
#             existing_entry = entry
#             break

#     if existing_entry:
#         # Update existing entry
#         existing_entry['Oversamp_factor'] = Oversamp_factor
#         existing_entry['rx_antenna_rad'] = rx_antenna_rad
#         existing_entry['azimuth_angle'] = azimuth_angle
#         existing_entry['mesh_angle_r'] = mesh_angle_r
#         existing_entry['mesh_angle_up'] = mesh_angle_up
#     else:
#         # Create and append a new entry if not found
#         new_entry = {
#             'Obj_range': Obj_range,
#             'RCS_without_const_dBsm': RCS_without_const_dBsm,
#             'Oversamp_factor': Oversamp_factor,
#             'rx_antenna_rad': rx_antenna_rad,
#             'azimuth_angle': azimuth_angle,
#             'mesh_angle_r':  mesh_angle_r,
#             'mesh_angle_up':  mesh_angle_up
#         }
#         data['Measurements'].append(new_entry)

#     # Write the updated or new data back to the file
#     with open(json_filename, 'w') as file:
#         json.dump(data, file, indent=4)
#     print(f"Data saved to respected file")

def save_to_json(Obj_range, mesh_angle_r, mesh_angle_up, RCS_with_const_dBsm, gain_factor, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle, elevation_angle,diff_const):
    # Initialize data structure with 'Measurements' key
    data = {'Measurements': []}

    # Attempt to open the file and read existing data
    if os.path.exists(json_filename) and os.path.getsize(json_filename) > 0:
      try:
         with open(json_filename, 'r') as file:
               existing_data = json.load(file)
               if 'Measurements' in existing_data:
                  data['Measurements'] = existing_data['Measurements']
               else:
                  print("No 'Measurements' key in JSON file. Initializing with empty list.")
      except (FileNotFoundError, json.JSONDecodeError):
         print(f"Error reading {json_filename}. Starting with an empty dataset.")

    # Create a new entry
    new_entry = {
        'Obj_range': Obj_range,
        'RCS_with_const_dBsm': RCS_with_const_dBsm,
        'Oversamp_factor': Oversamp_factor,
        'rx_antenna_rad': rx_antenna_rad,
        'azimuth_angle': azimuth_angle,
        'elevation_angle' :elevation_angle,
        'mesh_angle_r': mesh_angle_r,
        'mesh_angle_up': mesh_angle_up,
        'diff_const':diff_const,
        'gain_factor':gain_factor
    }
    data['Measurements'].append(new_entry)

    # Write the updated or new data back to the file
    with open(json_filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to {json_filename}")

def main():
   # normal mode
   # 0.1m
   # original from paper
   #comparison_image = simulate((800, 600), oversampling_factor=18, rx_radius=0.10, \
   #   number_sequences=120, enable_color_bar=True, enable_range_axis=False, filename="fft_image_campus_10cm_2.png")

   '''
   '''


   #Parameters for incident rays 
   image_width = 1200
   image_height = 600
   Oversamp_factor = 60
   Wavelength = 0.005

   # variables to change the range and theta
   Obj_range = 2
   iterations = 10
   range_azimuth = 2
   rx_antenna_rad = 0.3


   look_at_front =  np.array([0.0, 0, 0.0])
   vec_up = np.array([0.0, 0.0, 1.0]) 
   
   RCS_const = 0
   # Enter the type of the object (plate, sphere, corner)
   Object = "plate".lower()

   # adjusted for smaller gpus
   if Object == "human":
      render_mode_type = "RENDER_MODE_RAYTARGET_COMP"   #RENDER_MODE_RAYTARGET_COMP,  RENDER_MODE_GRAPHICS
      mesh_angle_r = 0
      mesh_angle_up = 0
      obj_width = 1.13
      obj_length = 1.85
      diff_const = 0.3
      RCS_const = 0.3164  #1.8440
      # From_vector = np.array([Obj_range, 0.0, 0.0])
      azimuth_angle =  calculate_azimuth_angle(obj_width, range_azimuth)
      elevation_angle =  calculate_elevation_angle(obj_length, range_azimuth)
      print(f"The azimuth angle is set to {azimuth_angle:.2f} degrees.")
      # azimuth_angle_rad = np.deg2rad(azimuth_angle)
      # simulate_sphere(image_width, image_height, Oversamp_factor, Wavelength, 
      #                 rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle,
      #                 mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type)
      
      script_directory = os.path.dirname(__file__)
      jsonfile_directory = os.path.join(script_directory, "../example-scripts/Avinash_json_files/Human_Json")
      json_filename = os.path.join(jsonfile_directory, "RCS_human_angle_results.json")
      for i in range(iterations):
         mesh_angle_r += 1
         From_vector = np.array([0.0, Obj_range, 0])
         print(f"The mesh_angle_r is {mesh_angle_r:.2f}")
         RCS_with_const_dBsm, gain_factor = simulate_human(image_width, image_height, Oversamp_factor, Wavelength, 
                      rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle,elevation_angle,
                      mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type, diff_const, RCS_const)

         time.sleep(4)
         save_to_json(Obj_range, mesh_angle_r, mesh_angle_up, RCS_with_const_dBsm, gain_factor, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle, elevation_angle, diff_const)



   # adjusted for smaller gpus
   elif Object == "sphere":
      render_mode_type = "RENDER_MODE_RAYTARGET_COMP"   #RENDER_MODE_RAYTARGET_COMP,  RENDER_MODE_GRAPHICS
      mesh_angle_r = 0
      mesh_angle_up = 0
      obj_width = 0.1
      obj_length = 0.1
      diff_const = 0.0
      RCS_const = 188.1575 #1.8440
      # From_vector = np.array([Obj_range, 0.0, 0.0])
      azimuth_angle =  calculate_azimuth_angle(obj_width, range_azimuth)
      elevation_angle =  calculate_elevation_angle(obj_length, range_azimuth)
      print(f"The azimuth angle is set to {azimuth_angle:.2f} degrees.")
      # azimuth_angle_rad = np.deg2rad(azimuth_angle)
      # simulate_sphere(image_width, image_height, Oversamp_factor, Wavelength, 
      #                 rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle,
      #                 mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type)
      
      script_directory = os.path.dirname(__file__)
      jsonfile_directory = os.path.join(script_directory, "../example-scripts/Avinash_json_files/Sphere_Json")
      json_filename = os.path.join(jsonfile_directory, "RCS_sphere_dist_results_auto.json")
      for i in range(iterations):
         Obj_range += 0
         From_vector = np.array([Obj_range, 0.0, 0.0])
         print(f"The Obj_range is {Obj_range:.2f}")
         RCS_without_const_dBsm = simulate_sphere(image_width, image_height, Oversamp_factor, Wavelength, 
                      rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle,elevation_angle,
                      mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type, diff_const, RCS_const)
         time.sleep(4)
         # save_to_json(Obj_range, mesh_angle_r, mesh_angle_up, RCS_without_const_dBsm, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle)

   
   elif Object == "plate":
      render_mode_type = "RENDER_MODE_RAYTARGET_COMP"   #RENDER_MODE_RAYTARGET_COMP,  RENDER_MODE_GRAPHICS
      obj_width = 0.05
      obj_length = 0.05
      mesh_angle_r = 0
      mesh_angle_up = 0
      diff_const = 0.0
      # rx_antenna_rad = 0.0
      RCS_const = 0.3164 

      
      azimuth_angle =  calculate_azimuth_angle(obj_width, range_azimuth)
      elevation_angle =  calculate_elevation_angle(obj_length, range_azimuth)

      
      # azimuth_angle_rad = np.deg2rad(azimuth_angle)
      # RCS_without_const_dBsm = simulate_plate(image_width, image_height, Oversamp_factor, Wavelength, 
      #                 rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle,
      #                 mesh_angle_r, Obj_range, render_mode_type)
      # azimuth_angle_rad = np.deg2rad(azimuth_angle)

      script_directory = os.path.dirname(__file__)
      jsonfile_directory = os.path.join(script_directory, "../example-scripts/Avinash_json_files/Plate_Json/")
      json_filename = os.path.join(jsonfile_directory, "RCS_plate_vs_azimuth_elevation.json")
      for i in range(iterations):
         azimuth_angle += 0
         elevation_angle += 0
         Obj_range += 5
         # rx_antenna_rad += 0.01
         From_vector = np.array([Obj_range, 0.0, 0.0])
         print(f"The rx_antenna_rad is {rx_antenna_rad:.2f}")
         print(f"The   Obj_range is {Obj_range:.2f}")
         print(f"The azimuth angle is set to {azimuth_angle:.2f} degrees.")
         print(f"The elevation angle is set to {elevation_angle:.2f} degrees.")
         RCS_with_const_dBsm, gain_factor = simulate_plate(image_width, image_height, Oversamp_factor, Wavelength, 
                      rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle, elevation_angle,
                      mesh_angle_r, mesh_angle_up,  Obj_range, render_mode_type, diff_const, RCS_const)
         time.sleep(3)
         # save_to_json(Obj_range, mesh_angle_r, mesh_angle_up, RCS_with_const_dBsm, gain_factor, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle, elevation_angle, diff_const)

   
   elif Object == "corner":
      render_mode_type = "RENDER_MODE_RAYTARGET_COMP"   #RENDER_MODE_RAYTARGET_COMP,  RENDER_MODE_GRAPHICS
      obj_width = 0.08
      obj_length = 0.08
      mesh_angle_r = 0
      mesh_angle_up = 45
      rx_antenna_rad = 0.3
      diff_const = 0.0
      RCS_const = 0.42465
      # From_vector = np.array([0.0, -(Obj_range), 0.0])
      script_directory = os.path.dirname(__file__)
      jsonfile_directory = os.path.join(script_directory, "../example-scripts/Avinash_json_files/Corner_Json/")
      json_filename = os.path.join(jsonfile_directory, "RCS_corner_vs_azimuth_elevation.json")
      
      
      azimuth_angle =  calculate_azimuth_angle(obj_width, range_azimuth)
      elevation_angle =  calculate_elevation_angle(obj_length, range_azimuth)
      
      #azimuth_angle_rad = np.deg2rad(azimuth_angle)
      # simulate_corner(image_width, image_height, Oversamp_factor, Wavelength, 
      #                 rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle,
      #                 mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type)
      
      for i in range(iterations):
         azimuth_angle += 1
         elevation_angle += 1
         print(f"The azimuth angle is set to {azimuth_angle:.2f} degrees.")
         print(f"The elevation angle is set to {elevation_angle:.2f} degrees.")
         # From_vector = np.array([0.0, -(Obj_range), 0.0])
         print(f"The rx_antenna_rad is {rx_antenna_rad:.2f}")
         From_vector = np.array([0.0, -(Obj_range), 0.0])
         RCS_with_const_dBsm, gain_factor = simulate_corner(image_width, image_height, Oversamp_factor, Wavelength, 
                       rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle, elevation_angle,
                       mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type, diff_const, RCS_const)
         time.sleep(3)
         save_to_json(Obj_range, mesh_angle_r, mesh_angle_up, RCS_with_const_dBsm, gain_factor, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle, elevation_angle, diff_const)

   else:
      print("Invalid Object type.")
      return
   
   # script_directory = os.path.dirname(__file__)
   # jsonfile_directory = os.path.join(script_directory, "../example-scripts/")
   # json_filename = os.path.join(jsonfile_directory, "RCS_plate_results_auto.json")
   # save_to_json(Obj_range, RCS_without_const_dBsm, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle)

if __name__ == "__main__":
   main()
   print("Finished")
































13-06-2024 (added human )

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

# Enter the type of the object (plate, sphere, corner)
Object = 0
graph_render_mode = 0


#Parameters for incident rays 
image_width = 0
image_height = 0
Oversamp_factor = 0
Wavelength = 0.0

# variables to change the range and theta
Obj_range = 0.0
range_rcs = 0.0
degree_interval = 0.0
rx_antenna_rad = 0.0
mesh_angle_r = 0
mesh_angle_up = 0
diff_const = 0
RCS_const = 0
RCS_with_const = 0


azimuth_angle = 0
look_at_front =  np.array([0.0, 0, 0.0])
vec_up = np.array([0.0, 0.0, 0.0])
# RCS_constant = 0.0


def calculate_coordinates(radius, degree):
   angle_rad = np.deg2rad(degree)
   # Define the named tuple format

   # Calculate x and y coordinates in meters
   From_x = np.float32(radius * np.cos(angle_rad))
   From_y = np.float32(radius * np.sin(angle_rad))
   return (From_x, From_y)

def plot_rms_and_std(error_array):
   x = np.array(range(error_array[:,0].shape[0]))
   y = error_array[:,0]
   e = error_array[:,1]
   plt.errorbar(x, y, e, linestyle='None', marker='^')

   plt.savefig("error_plot.png")

def plot_sphere_radius_performance():
   """
   method for paper plot
   """
   mpl.rc('font',family='Times New Roman')
   # gold-standard: 284sek # arround 11000 rays
   sphere_sizes = [0.5, 1.0, 2.0, 4.0]
   elapsed_sphere_times = [12.24, 4.0, 1.17, 0.43]
   elapsed_sphere_times_fast = [5.44, 1.96, 0.56, 0.28]

   elapsed_cone_times = [4.0, 2.0, 0.8, 0.71]
   elapsed_cone_times_fast = [2.23, 1.14, 0.52, 0.43]

   plt.plot(sphere_sizes, elapsed_sphere_times, marker='o', label="vary sphere size")
   plt.plot(sphere_sizes, elapsed_sphere_times_fast, marker='x', label="vary sphere size optimized")
   plt.plot(sphere_sizes, elapsed_cone_times, marker='o', label="vary cone angle")
   plt.plot(sphere_sizes, elapsed_cone_times_fast, marker='x', label="vary cone angle optimized")
   ax = plt.gca()
   ax.set_xticks(sphere_sizes)
   ax.set_xticklabels(['0.5/2°', '1.0/4.0°', '2.0/8°' ,'4.0/16°'])

   plt.legend(prop={"size":14})
   plt.xticks(fontsize=14)
   plt.yticks(fontsize=14)
   plt.grid()
   plt.xlabel("Sphere radius (m)/Cone Angle (deg.)", size=14)
   plt.ylabel("Time in s", size=14)
   plt.savefig("SphereSizePerformance.pdf", bbox_inches='tight')

def plot_antenna_configuration(radar_signal_data):

   mpl.rc('font',family='Times New Roman')
   tx_positions_2d = radar_signal_data.tx_positions[:, :]
   rx_positions_2d = radar_signal_data.rx_positions[:, :]

   tx_positions_2d -= tx_positions_2d[0]
   rx_positions_2d -= rx_positions_2d[0]
   csfont = {'fontname':'Times New Roman'}

   plt.scatter(rx_positions_2d[:, 0], rx_positions_2d[:, 1], marker='o', label="rx antenna positions")
   ax = plt.gca()
   ax.set_yticklabels([])
   ax.set_yticks([])
   ax.set_xlabel("x in m", size=14, **csfont)
   plt.xticks(**csfont)

   ax.scatter(tx_positions_2d[:, 0], tx_positions_2d[:, 1], marker='x', label="tx antenna positions")

   virtual_positions = []
   for rx_pos in rx_positions_2d:
      for tx_pos in tx_positions_2d:
         virtual_pos = (rx_pos + tx_pos)
         virtual_positions.append(virtual_pos)

   virtual_positions = np.asarray(virtual_positions)
   
   ax.scatter(virtual_positions[:, 0], virtual_positions[:, 1]-0.1, marker='o', color='blue', label="virtual antenna positions")
   ax.legend()
   ax.set_aspect(0.33)
   plt.savefig("antenna_positions.eps", bbox_inches='tight')


def get_max_range(radar_signal_data):
   c=3e8
   chirp_duration = radar_signal_data.chirp_duration
   bandwidth = radar_signal_data.bandwidth
   sample_frequency = radar_signal_data.time_vector.shape[0]/chirp_duration
   r_max = (c*chirp_duration*sample_frequency)/(4*bandwidth)*2 # complex signal multiply by 2

   return r_max

def compute_azimuth_label_sine_space(number_sample_points, angular_dim, start_sin_index=0, stop_sin_index=None):
   
   if not stop_sin_index:
      stop_sin_index = angular_dim

   start_sin_value = -2.0/angular_dim*start_sin_index + 1
   end_sin_value = -2.0/angular_dim*stop_sin_index + 1
   #sin_space_labels = np.linspace(1, -1, number_sample_points)

   sin_space_labels = np.linspace(start_sin_value, end_sin_value, number_sample_points)
   angular_labels = np.round(np.rad2deg(np.arcsin(sin_space_labels))).astype(np.int32)
   angular_positions = np.linspace(0, angular_dim-1, number_sample_points)

   #angular_labels = np.round(np.linspace(90, -90, number_sample_points))
   #angular_positions = (np.sin(np.deg2rad(angular_labels)) + 1.0)*0.5 * (angular_dim-1)
   return angular_positions, angular_labels

def compute_range_label(radar_signal_data, number_sample_points, range_dim):
   r_max = get_max_range(radar_signal_data)
   print("r_max: " + str(r_max))
   range_labels = np.round(np.linspace(0, r_max, number_sample_points)).astype(np.int32)
   range_positions = np.linspace(0, range_dim, number_sample_points)
   return range_positions, range_labels

def load_Object(renderer, material_dir, obj_filename, mesh_angle_r, mesh_angle_up, diff_const):
   mesh_list, obj_mat_list = load_mesh_normal_from_obj(obj_filename, material_dir)

   for i, obj_mat in enumerate(obj_mat_list):
      mesh = mesh_list[i]
      if "metal" in obj_mat.name.lower():
         mesh_mat = MaterialMixed(obj_mat.diffuse, diff_const)
      else:
         #mesh_mat = MaterialMetal(obj_mat.diffuse, 0.1)
         mesh_mat = MaterialMixed(obj_mat.diffuse, 0.1)
      print("Mesh loading completed")

      mesh.set_material(mesh_mat)
      mesh.rotate([1.0, 0.0, 0.0],np.deg2rad(mesh_angle_up))
      mesh.rotate([0.0, 0.0, 1.0],np.deg2rad(mesh_angle_r))
      renderer.add_geometry_object(mesh)


def calculate_azimuth_angle(obj_width, range_rcs):
    # Calculate the half-angle theta/2 in radians
    theta_half_radians = np.arctan((obj_width / 2) / range_rcs)
    
    # Calculate the full theta in radians
    azimuth_radians = 2 * theta_half_radians
    
    # Convert theta from radians to degrees
    azimuth_angle = np.degrees(azimuth_radians)
    
    return azimuth_angle

def calculate_elevation_angle(obj_length, range_rcs):
    # Calculate the half-angle theta/2 in radians
    theta_half_radians = np.arctan((obj_length / 2) / range_rcs)
    
    # Calculate the full theta in radians
    elevation_radians = 2 * theta_half_radians
    
    # Convert theta from radians to degrees
    elevation_angle = np.degrees(elevation_radians)
    
    return elevation_angle

      # Define the integrand function
def integrand(theta, phi):
   # Use the single RCS value directly in linear scale
   rcs_value = RCS_with_const 
   # Return the product of RCS and cosine of the elevation angle
   # print(f"rcs_value {rcs_value} rcs_value inside the integrand")
   return rcs_value * np.cos(theta)


def load_antennas_for_imaging_iwr6843AOP(render_pos, look_at_front, vec_up,
                                         radiation_pattern, phi_axis, theta_axis, Wavelength, rx_antenna_rad, azimuth_angle, elevation_angle):

   tx_antennas = list()
   rx_antennas = list()

   #construct the antenna array
   num_tx = 3
   num_rx = 4
   tx_antenna_pos_offset_1 = np.array([-Wavelength,0,0])
   tx_antenna_pos_offset_2 = np.array([0,-Wavelength,0])
   tx_antenna_pos_offset_3 = np.array([0,0,0])
   tx_offsets = [tx_antenna_pos_offset_1, tx_antenna_pos_offset_2, tx_antenna_pos_offset_3]
   rx_antenna_pos_offset_1 = np.array([-Wavelength,-1.5*Wavelength,0])
   rx_antenna_pos_offset_2 = np.array([-1.5*Wavelength,-1.5*Wavelength,0])
   rx_antenna_pos_offset_3 = np.array([-1.5*Wavelength,-Wavelength,0])
   rx_antenna_pos_offset_4 = np.array([-Wavelength,-Wavelength,0])
   rx_offsets = [rx_antenna_pos_offset_1, rx_antenna_pos_offset_2, rx_antenna_pos_offset_3, rx_antenna_pos_offset_4]

   for i in range(num_tx):
      tx_antenna_pos = render_pos + tx_offsets[i] #for i= 0 [0.2951,0,0]
      #tx_antenna_pos = tx_antennas_pos[i] + camera_pos_front
      tx_antenna = TxAntenna(tx_antenna_pos)
      tx_antenna.set_look_at(look_at_front)
      tx_antenna.set_up(vec_up)
      tx_antenna.set_azimuth(np.deg2rad(azimuth_angle))
      tx_antenna.set_elevation(np.deg2rad(elevation_angle))
      tx_antenna.set_radiation_pattern(radiation_pattern, phi_axis, theta_axis)
      tx_antennas.append(tx_antenna)
         
   # add rx antenna pos 
   for i in range(num_rx):
      rx_antenna_pos = render_pos + rx_offsets[i]
      rx_antenna = RxAntenna(rx_antenna_pos, rx_antenna_rad)
      rx_antenna.set_look_at(look_at_front)
      rx_antenna.set_up(vec_up)
      rx_antenna.set_radiation_pattern(radiation_pattern, phi_axis, theta_axis)
      rx_antennas.append(rx_antenna)

   return tx_antennas, rx_antennas

def simulate_plate(image_width, image_height, Oversamp_factor, Wavelength, 
                      rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle, elevation_angle,
                      mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type, diff_const, RCS_const):
   renderer = Renderer()
   render_mode = getattr(RenderMode, render_mode_type)
   
   # load mesh from outside

   script_directory = os.path.dirname(__file__)
   content_directory = os.path.join(script_directory, "../example-files/Avinash_RCS/Plate/")
   obj_filename = os.path.join(content_directory, "Plate_high_roughness_with_thickness.obj")#Plate_high_roughness_metal, Plate_high_roughness_flipped
   #print("start loading campus scene")
   load_Object(renderer, content_directory, obj_filename, mesh_angle_r, mesh_angle_up, diff_const)

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
      print("Image stored in RCS_cube_plate")
      # image_array.save("RCS_corner_multipath.png")
      # Ensure the directory exists
      os.makedirs(render_directory, exist_ok=True)

      # Define the full path to the image file
      image_path = os.path.join(render_directory, "RCS_cube_plate.png")

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

      azimuth_radi = np.deg2rad(azimuth_angle)
      elevation_radi = np.deg2rad(elevation_angle)

      # For the effective area, performing double integral over azimuth and elevation, which provides effective area in the omni directional sphere 
      effective_area, error = dblquad(lambda phi, theta: np.cos(theta),
                                -azimuth_radi / 2, azimuth_radi / 2,  # Azimuth angle range
                                lambda x: -elevation_radi / 2, lambda x: elevation_radi / 2)  # Elevation angle range

      print(f"effective_area is: {effective_area:.4f}")
      # To calculate the gain factor
      gain_factor = (4 * np.pi) / effective_area
      print(f"gain_factor_dBsm : {gain_factor:.4f}")
      gain_factor_dBsm = 10 * np.log10(gain_factor)

      print(f"gain_factor_dBsm : {gain_factor_dBsm:.4f}")

      # RCS_with_const_gain = RCS_with_const / gain_factor

      RCS_with_const_dBsm = 10 * np.log10(RCS_with_const) 

      RCS_with_const_gain_dbsm = RCS_with_const_dBsm - gain_factor_dBsm

      print(f" RCS in dBsm {RCS_with_const_gain_dbsm}") 

   
      return RCS_with_const_gain_dbsm, gain_factor_dBsm






def simulate_corner_multipath(image_width, image_height, Oversamp_factor, Wavelength, 
                      rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle, elevation_angle,
                      mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type, diff_const, RCS_const):

   renderer = Renderer()
   render_mode = getattr(RenderMode, render_mode_type)
   
   # load mesh from outside

   script_directory = os.path.dirname(__file__)
   content_directory = os.path.join(script_directory, "../example-files/Avinash_RCS/Corner_multipath/")#
   obj_filename = os.path.join(content_directory, "Corner_multipath_05cm.obj")#Corner_MOM24_mesh,Corner_MOM24_mesh, Corner_MOM24_withmetal , Corner_7cm
   #print("start loading campus scene")
   load_Object(renderer, content_directory, obj_filename, mesh_angle_r, mesh_angle_up, diff_const)

   radiation_pattern_filename = os.path.join(script_directory, "../example-files/Radiation_pattern_new.txt")
   #radiation_pattern_filename = os.path.join(script_directory, "../example-files/test_radiation_pattern.txt")
   radiation_pattern, phi_axis, theta_axis = load_radiation_pattern_from_cst(radiation_pattern_filename)
   #plot_radiation_pattern_3d(radiation_pattern, phi_axis, theta_axis, subsampling_factor=8)
   #plt.show()

   
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
      print("Image stored in RCS_corner_multipath")
      # image_array.save("RCS_corner_multipath.png")
      # Ensure the directory exists
      os.makedirs(render_directory, exist_ok=True)

      # Define the full path to the image file
      image_path = os.path.join(render_directory, "RCS_corner_multipath.png")

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


      RCS_without_const = Power_ratio*(4*np.pi*((Obj_range)**2))**2
      
      RCS_with_const = RCS_without_const*RCS_const

      print(f"RCS_with_const {RCS_with_const}")

      azimuth_radi = np.deg2rad(azimuth_angle)
      elevation_radi = np.deg2rad(elevation_angle)

      #  Performing the double integral over the azimuth and elevation
      effective_area, error = dblquad(lambda phi,theta:  np.cos(theta),
                                -azimuth_radi / 2, azimuth_radi / 2,  # Azimuth angle range
                                lambda x: -elevation_radi / 2, lambda x: elevation_radi / 2)  # Elevation angle range

      # Calculate the gain factor
      gain_factor = (4 * np.pi) / effective_area

      print(f"gain_factor : {gain_factor:.4f}")

      RCS_with_const_dBsm = 10 * np.log10(RCS_with_const)

      print(f" RCS in dBsm {RCS_with_const_dBsm}") 

      return RCS_with_const_dBsm, gain_factor


def simulate_corner(image_width, image_height, Oversamp_factor, Wavelength, 
                      rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle, elevation_angle,
                      mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type, diff_const, RCS_const):

   renderer = Renderer()
   render_mode = getattr(RenderMode, render_mode_type)
   
   # load mesh from outside

   script_directory = os.path.dirname(__file__)
   content_directory = os.path.join(script_directory, "../example-files/Avinash_RCS/Corner/")#
   obj_filename = os.path.join(content_directory, "Corner_15cm.obj")#Corner_MOM24_mesh,Corner_MOM24_mesh, Corner_MOM24_withmetal , Corner_7cm
   #print("start loading campus scene")
   load_Object(renderer, content_directory, obj_filename, mesh_angle_r, mesh_angle_up, diff_const)

   radiation_pattern_filename = os.path.join(script_directory, "../example-files/Radiation_pattern_new.txt")
   #radiation_pattern_filename = os.path.join(script_directory, "../example-files/test_radiation_pattern.txt")
   radiation_pattern, phi_axis, theta_axis = load_radiation_pattern_from_cst(radiation_pattern_filename)
   #plot_radiation_pattern_3d(radiation_pattern, phi_axis, theta_axis, subsampling_factor=8)
   #plt.show()

   
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
      print("Image stored in RCS_corner")
      # image_array.save("RCS_corner_multipath.png")
      # Ensure the directory exists
      os.makedirs(render_directory, exist_ok=True)

      # Define the full path to the image file
      image_path = os.path.join(render_directory, "RCS_corner.png")

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


      RCS_without_const = Power_ratio*(4*np.pi*((Obj_range)**2))**2
      
      RCS_with_const = RCS_without_const*RCS_const

      print(f"RCS_with_const {RCS_with_const}")

      azimuth_radi = np.deg2rad(azimuth_angle)
      elevation_radi = np.deg2rad(elevation_angle)

      #  Performing the double integral over the azimuth and elevation
      effective_area, error = dblquad(lambda phi,theta:  np.cos(theta),
                                -azimuth_radi / 2, azimuth_radi / 2,  # Azimuth angle range
                                lambda x: -elevation_radi / 2, lambda x: elevation_radi / 2)  # Elevation angle range

      # Calculate the gain factor
      gain_factor = (4 * np.pi) / effective_area

      print(f"gain_factor : {gain_factor:.4f}")

      RCS_with_const_dBsm = 10 * np.log10(RCS_with_const)

      print(f" RCS in dBsm {RCS_with_const_dBsm}") 

      return RCS_with_const_dBsm, gain_factor


def simulate_sphere(image_width, image_height, Oversamp_factor, Wavelength, 
                     rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle_rad,elevation_angle,
                     mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type, diff_const, RCS_const):

   renderer = Renderer()
   render_mode = getattr(RenderMode, render_mode_type)
   print(f"The diff_const is {diff_const}")
   # load mesh from outside

   script_directory = os.path.dirname(__file__)
   content_directory = os.path.join(script_directory, "../example-files/Avinash_RCS/Sphere/")
   obj_filename = os.path.join(content_directory, "sphere_15cm.obj")#Avinash_sphere, Spehere_RCS_metal
   #print("start loading campus scene")
   load_Object(renderer, content_directory, obj_filename, mesh_angle_r, mesh_angle_up, diff_const)

   radiation_pattern_filename = os.path.join(script_directory, "../example-files/Radiation_pattern_new.txt")
   #radiation_pattern_filename = os.path.join(script_directory, "../example-files/test_radiation_pattern.txt")
   radiation_pattern, phi_axis, theta_axis = load_radiation_pattern_from_cst(radiation_pattern_filename)
   plot_radiation_pattern_3d(radiation_pattern, phi_axis, theta_axis)
   plt.show()

   # set tx and rx antennas
   # antenna_offset = np.array([-8.8, 32.0, 0.7])

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
      print("Image stored in RCS_sphere_5cm")
      image_array.save("RCS_sphere_5cm.png")
   elif render_mode == RenderMode.RENDER_MODE_RAYTARGET_COMP:
      # This mode supports accurate Doppler and Radiation-Patterns
      
      tx_antennas, rx_antennas = load_antennas_for_imaging_iwr6843AOP(From_vector, look_at_front, vec_up, radiation_pattern, phi_axis, theta_axis, Wavelength, rx_antenna_rad, azimuth_angle_rad, elevation_angle)
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
      print(f" RCS without const factor linear {RCS_without_const}")
      RCS_with_const = RCS_without_const * RCS_const
      RCS_with_const_dBsm = 10 * np.log10(RCS_with_const) 
      print(f" RCS with const factor dBsm {RCS_with_const_dBsm}") 
      return RCS_with_const_dBsm
      # Converting gains from dB to linear scale
      # RCS_final = (RCS_without_const)
      # print(f" RCS with const linear {RCS_final}")
      # # print(f" RCS with const factor linear {RCS_final}") 
      # RCS_dBsm = 10 * np.log10(RCS_final) 

      # print(f" RCS with const factor dBsm {RCS_dBsm}")


def simulate_human(image_width, image_height, Oversamp_factor, Wavelength, 
                     rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle_rad,elevation_angle,
                     mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type, diff_const, RCS_const):

   renderer = Renderer()
   render_mode = getattr(RenderMode, render_mode_type)
   print(f"The diff_const is {diff_const}")
   # load mesh from outside

   script_directory = os.path.dirname(__file__)
   content_directory = os.path.join(script_directory, "../example-files/Avinash_RCS/Human/")
   obj_filename = os.path.join(content_directory, "Peter_RCS.obj")#Avinash_sphere, Spehere_RCS_metal
   #print("start loading campus scene")
   load_Object(renderer, content_directory, obj_filename, mesh_angle_r, mesh_angle_up, diff_const)

   radiation_pattern_filename = os.path.join(script_directory, "../example-files/Radiation_pattern_new.txt")
   #radiation_pattern_filename = os.path.join(script_directory, "../example-files/test_radiation_pattern.txt")
   radiation_pattern, phi_axis, theta_axis = load_radiation_pattern_from_cst(radiation_pattern_filename)
   # plot_radiation_pattern_3d(radiation_pattern, phi_axis, theta_axis)
   # plt.show()

   # set tx and rx antennas
   # antenna_offset = np.array([-8.8, 32.0, 0.7])

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
      # print("Image stored in RCS_human")
      # image_array.save("RCS_human.png")



      render_directory = os.path.join(script_directory, "../example-scripts/Avinash_render_images/")
      print("Image stored in RCS_human")
      # image_array.save("RCS_corner_multipath.png")
      # Ensure the directory exists
      os.makedirs(render_directory, exist_ok=True)

      # Define the full path to the image file
      image_path = os.path.join(render_directory, "RCS_human.png")

      # Save the image to the specified directory
      image_array.save(image_path)



   elif render_mode == RenderMode.RENDER_MODE_RAYTARGET_COMP:
      # This mode supports accurate Doppler and Radiation-Patterns
      
      tx_antennas, rx_antennas = load_antennas_for_imaging_iwr6843AOP(From_vector, look_at_front, vec_up, radiation_pattern, phi_axis, theta_axis, Wavelength, rx_antenna_rad, azimuth_angle_rad, elevation_angle)
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
      print(f" RCS without const factor linear {RCS_without_const}")
      RCS_with_const = RCS_without_const * RCS_const
      RCS_with_const_dBsm = 10 * np.log10(RCS_with_const) 
      print(f" RCS with const factor dBsm {RCS_with_const_dBsm}") 
      gain_factor = 1
      return RCS_with_const_dBsm, gain_factor

# def save_to_json(Obj_range, mesh_angle_r, mesh_angle_up, RCS_without_const_dBsm, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle):
#     # Initialize data structure with 'Measurements' key
#     data = {'Measurements': []}

#     # Load existing data if the file exists and is not empty
#     if os.path.exists(json_filename) and os.path.getsize(json_filename) > 0:
#         with open(json_filename, 'r') as file:
#             try:
#                 existing_data = json.load(file)
#                 if 'Measurements' in existing_data:
#                     data['Measurements'] = existing_data['Measurements']
#                 else:
#                     print("No 'Measurements' key in JSON file. Initializing with empty list.")
#             except json.JSONDecodeError:
#                 print(f"Error decoding JSON from {json_filename}. Starting with an empty dataset.")

#     # Check if this specific measurement already exists
#     existing_entry = None
#     for entry in data['Measurements']:
#         if (entry['Obj_range'] == Obj_range and entry['RCS_without_const_dBsm'] == RCS_without_const_dBsm):
#             existing_entry = entry
#             break

#     if existing_entry:
#         # Update existing entry
#         existing_entry['Oversamp_factor'] = Oversamp_factor
#         existing_entry['rx_antenna_rad'] = rx_antenna_rad
#         existing_entry['azimuth_angle'] = azimuth_angle
#         existing_entry['mesh_angle_r'] = mesh_angle_r
#         existing_entry['mesh_angle_up'] = mesh_angle_up
#     else:
#         # Create and append a new entry if not found
#         new_entry = {
#             'Obj_range': Obj_range,
#             'RCS_without_const_dBsm': RCS_without_const_dBsm,
#             'Oversamp_factor': Oversamp_factor,
#             'rx_antenna_rad': rx_antenna_rad,
#             'azimuth_angle': azimuth_angle,
#             'mesh_angle_r':  mesh_angle_r,
#             'mesh_angle_up':  mesh_angle_up
#         }
#         data['Measurements'].append(new_entry)

#     # Write the updated or new data back to the file
#     with open(json_filename, 'w') as file:
#         json.dump(data, file, indent=4)
#     print(f"Data saved to respected file")

def save_to_json(Obj_range, mesh_angle_r, mesh_angle_up, RCS_with_const_dBsm, gain_factor, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle, elevation_angle,diff_const, specification):
    # Initialize data structure with 'Measurements' key
    data = {'Measurements': []}

    # Attempt to open the file and read existing data
    if os.path.exists(json_filename) and os.path.getsize(json_filename) > 0:
      try:
         with open(json_filename, 'r') as file:
               existing_data = json.load(file)
               if 'Measurements' in existing_data:
                  data['Measurements'] = existing_data['Measurements']
               else:
                  print("No 'Measurements' key in JSON file. Initializing with empty list.")
      except (FileNotFoundError, json.JSONDecodeError):
         print(f"Error reading {json_filename}. Starting with an empty dataset.")

    # Create a new entry
    new_entry = {
        'Specification': specification,
        'Obj_range': Obj_range,
        'RCS_with_const_dBsm': RCS_with_const_dBsm,
        'Oversamp_factor': Oversamp_factor,
        'rx_antenna_rad': rx_antenna_rad,
        'azimuth_angle': azimuth_angle,
        'elevation_angle' :elevation_angle,
        'mesh_angle_r': mesh_angle_r,
        'mesh_angle_up': mesh_angle_up,
        'diff_const':diff_const,
        'gain_factor':gain_factor
    }
    data['Measurements'].append(new_entry)

    # Write the updated or new data back to the file
    with open(json_filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to {json_filename}")

def main():
   # normal mode
   # 0.1m
   # original from paper
   #comparison_image = simulate((800, 600), oversampling_factor=18, rx_radius=0.10, \
   #   number_sequences=120, enable_color_bar=True, enable_range_axis=False, filename="fft_image_campus_10cm_2.png")

   '''
   '''


   #Parameters for incident rays 
   image_width = 1200
   image_height = 600
   Oversamp_factor = 60
   Wavelength = 0.005

   # variables to change the range and theta
   Obj_range = 2

   iterations = 1
   range_azimuth = 2
   rx_antenna_rad = 0.3


   look_at_front =  np.array([0.0, 0, 0.0])
   vec_up = np.array([0.0, 0.0, 1.0]) 
   
   RCS_const = 0
   # Enter the type of the object (plate, sphere, corner)
   Object = "plate".lower()

   # adjusted for smaller gpus
   if Object == "human":
      render_mode_type = "RENDER_MODE_RAYTARGET_COMP"   #RENDER_MODE_RAYTARGET_COMP,  RENDER_MODE_GRAPHICS
      mesh_angle_r = 0
      mesh_angle_up = 0
      obj_width = 1.13
      obj_length = 1.85
      diff_const = 0.3
      RCS_const = 0.3164  #1.8440
      # From_vector = np.array([Obj_range, 0.0, 0.0])
      azimuth_angle =  calculate_azimuth_angle(obj_width, range_azimuth)
      elevation_angle =  calculate_elevation_angle(obj_length, range_azimuth)
      print(f"The azimuth angle is set to {azimuth_angle:.2f} degrees.")
      # azimuth_angle_rad = np.deg2rad(azimuth_angle)
      # simulate_sphere(image_width, image_height, Oversamp_factor, Wavelength, 
      #                 rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle,
      #                 mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type)
      
      script_directory = os.path.dirname(__file__)
      jsonfile_directory = os.path.join(script_directory, "../example-scripts/Avinash_json_files/Human_Json")
      json_filename = os.path.join(jsonfile_directory, "RCS_human_angle_results.json")
      for i in range(iterations):
         mesh_angle_r += 0
         From_vector = np.array([0.0, Obj_range, 0])
         print(f"The mesh_angle_r is {mesh_angle_r:.2f}")
         RCS_with_const_dBsm, gain_factor = simulate_human(image_width, image_height, Oversamp_factor, Wavelength, 
                      rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle,elevation_angle,
                      mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type, diff_const, RCS_const)

         time.sleep(4)
         save_to_json(Obj_range, mesh_angle_r, mesh_angle_up, RCS_with_const_dBsm, gain_factor, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle, elevation_angle, diff_const, specification)



   # adjusted for smaller gpus
   elif Object == "sphere":
      render_mode_type = "RENDER_MODE_RAYTARGET_COMP"   #RENDER_MODE_RAYTARGET_COMP,  RENDER_MODE_GRAPHICS
      mesh_angle_r = 0
      mesh_angle_up = 0
      obj_width = 0.1
      obj_length = 0.1
      diff_const = 0.0
      RCS_const = 188.1575 #1.8440
      # From_vector = np.array([Obj_range, 0.0, 0.0])
      azimuth_angle =  calculate_azimuth_angle(obj_width, range_azimuth)
      elevation_angle =  calculate_elevation_angle(obj_length, range_azimuth)
      print(f"The azimuth angle is set to {azimuth_angle:.2f} degrees.")
      # azimuth_angle_rad = np.deg2rad(azimuth_angle)
      # simulate_sphere(image_width, image_height, Oversamp_factor, Wavelength, 
      #                 rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle,
      #                 mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type)
      
      script_directory = os.path.dirname(__file__)
      jsonfile_directory = os.path.join(script_directory, "../example-scripts/Avinash_json_files/Sphere_Json")
      json_filename = os.path.join(jsonfile_directory, "RCS_sphere_dist_results_auto.json")
      for i in range(iterations):
         Obj_range += 0
         From_vector = np.array([Obj_range, 0.0, 0.0])
         print(f"The Obj_range is {Obj_range:.2f}")
         RCS_without_const_dBsm = simulate_sphere(image_width, image_height, Oversamp_factor, Wavelength, 
                      rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle,elevation_angle,
                      mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type, diff_const, RCS_const)
         time.sleep(4)
         # save_to_json(Obj_range, mesh_angle_r, mesh_angle_up, RCS_without_const_dBsm, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle)

   
   elif Object == "plate":
      render_mode_type = "RENDER_MODE_RAYTARGET_COMP"   #RENDER_MODE_RAYTARGET_COMP,  RENDER_MODE_GRAPHICS
      obj_width = 0.05
      obj_length = 0.05
      mesh_angle_r = 0
      mesh_angle_up = 85
      diff_const = 0.7
      # rx_antenna_rad = 0.0
      RCS_const = 0.3164 
      specification= "RCS vs angle for normal plate uprotation"

      
      azimuth_angle =  calculate_azimuth_angle(obj_width, range_azimuth)
      elevation_angle =  calculate_elevation_angle(obj_length, range_azimuth)

      
      # azimuth_angle_rad = np.deg2rad(azimuth_angle)
      # RCS_without_const_dBsm = simulate_plate(image_width, image_height, Oversamp_factor, Wavelength, 
      #                 rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle,
      #                 mesh_angle_r, Obj_range, render_mode_type)
      # azimuth_angle_rad = np.deg2rad(azimuth_angle)

      script_directory = os.path.dirname(__file__)
      jsonfile_directory = os.path.join(script_directory, "../example-scripts/Avinash_json_files/Plate_Json/")
      json_filename = os.path.join(jsonfile_directory, "RCS_plate_angle_results_auto.json")
      for i in range(iterations):
         mesh_angle_up += 0
      
         # rx_antenna_rad += 0.01
         From_vector = np.array([Obj_range, 0.0, 0.0])
         print(f"The   Obj_range is {Obj_range:.2f}")
         print(f"The mesh_angle_up angle is set to {mesh_angle_up:.2f} degrees.")
         RCS_with_const_dBsm, gain_factor = simulate_plate(image_width, image_height, Oversamp_factor, Wavelength, 
                      rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle, elevation_angle,
                      mesh_angle_r, mesh_angle_up,  Obj_range, render_mode_type, diff_const, RCS_const)
         time.sleep(3)
         #  
         save_to_json(Obj_range, mesh_angle_r, mesh_angle_up, RCS_with_const_dBsm, gain_factor, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle, elevation_angle, diff_const, specification)



   elif Object == "corner_multipath":

      render_mode_type = "RENDER_MODE_RAYTARGET_COMP"   #RENDER_MODE_RAYTARGET_COMP,  RENDER_MODE_GRAPHICS
      obj_width = 0.08
      obj_length = 0.08
      mesh_angle_r = 0
      mesh_angle_up = 0
      rx_antenna_rad = 0.3
      diff_const = 0.0
      RCS_const = 0.42465
      specification = "RCS corner multipath with ground"
      # From_vector = np.array([0.0, -(Obj_range), 0.0])
      script_directory = os.path.dirname(__file__)
      jsonfile_directory = os.path.join(script_directory, "../example-scripts/Avinash_json_files/Corner_multipath_Json/")
      json_filename = os.path.join(jsonfile_directory, "RCS_corner_multipath_dist.json")
      
      
      azimuth_angle =  calculate_azimuth_angle(obj_width, range_azimuth)
      elevation_angle =  calculate_elevation_angle(obj_length, range_azimuth)
      
      #azimuth_angle_rad = np.deg2rad(azimuth_angle)
      # simulate_corner(image_width, image_height, Oversamp_factor, Wavelength, 
      #                 rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle,
      #                 mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type)
      
      for i in range(iterations):
         Obj_range += 1
         # azimuth_angle += 1
         # elevation_angle += 1
         print(f"The Obj_range is set to {Obj_range:.2f}")
         print(f"The azimuth angle is set to {azimuth_angle:.2f} degrees.")
         print(f"The elevation angle is set to {elevation_angle:.2f} degrees.")
         # From_vector = np.array([0.0, -(Obj_range), 0.0])
         print(f"The rx_antenna_rad is {rx_antenna_rad:.2f}")
         From_vector = np.array([0.0, -(Obj_range), 0.0])
         RCS_with_const_dBsm, gain_factor = simulate_corner_multipath(image_width, image_height, Oversamp_factor, Wavelength, 
                       rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle, elevation_angle,
                       mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type, diff_const, RCS_const)
         time.sleep(3)
         save_to_json(Obj_range, mesh_angle_r, mesh_angle_up, RCS_with_const_dBsm, gain_factor, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle, elevation_angle, diff_const, specification)



   elif Object == "corner":
      render_mode_type = "RENDER_MODE_RAYTARGET_COMP"   #RENDER_MODE_RAYTARGET_COMP,  RENDER_MODE_GRAPHICS
      obj_width = 0.08
      obj_length = 0.08
      mesh_angle_r = 0
      mesh_angle_up = 45
      rx_antenna_rad = 0.3
      diff_const = 0.0
      RCS_const = 0.42465
      # From_vector = np.array([0.0, -(Obj_range), 0.0])
      script_directory = os.path.dirname(__file__)
      jsonfile_directory = os.path.join(script_directory, "../example-scripts/Avinash_json_files/Corner_Json/")
      json_filename = os.path.join(jsonfile_directory, "RCS_corner_vs_azimuth_elevation.json")
      
      
      azimuth_angle =  calculate_azimuth_angle(obj_width, range_azimuth)
      elevation_angle =  calculate_elevation_angle(obj_length, range_azimuth)
      
      #azimuth_angle_rad = np.deg2rad(azimuth_angle)
      # simulate_corner(image_width, image_height, Oversamp_factor, Wavelength, 
      #                 rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle,
      #                 mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type)
      
      for i in range(iterations):
         azimuth_angle += 1
         elevation_angle += 1
         print(f"The azimuth angle is set to {azimuth_angle:.2f} degrees.")
         print(f"The elevation angle is set to {elevation_angle:.2f} degrees.")
         # From_vector = np.array([0.0, -(Obj_range), 0.0])
         print(f"The rx_antenna_rad is {rx_antenna_rad:.2f}")
         From_vector = np.array([0.0, -(Obj_range), 0.0])
         RCS_with_const_dBsm, gain_factor = simulate_corner(image_width, image_height, Oversamp_factor, Wavelength, 
                       rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle, elevation_angle,
                       mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type, diff_const, RCS_const)
         time.sleep(3)
         save_to_json(Obj_range, mesh_angle_r, mesh_angle_up, RCS_with_const_dBsm, gain_factor, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle, elevation_angle, diff_const, specification)

   else:
      print("Invalid Object type.")
      return
   
   # script_directory = os.path.dirname(__file__)
   # jsonfile_directory = os.path.join(script_directory, "../example-scripts/")
   # json_filename = os.path.join(jsonfile_directory, "RCS_plate_results_auto.json")
   # save_to_json(Obj_range, RCS_without_const_dBsm, json_filename, Oversamp_factor, rx_antenna_rad, azimuth_angle)

if __name__ == "__main__":
   main()
   print("Finished")



