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
from radar_ray_python.Mesh import Mesh
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

from Avinash_RCS import calculate_azimuth_angle, calculate_elevation_angle, load_antennas_for_imaging_iwr6843AOP, load_Object

def simulate_plate(image_width, image_height, Oversamp_factor, Wavelength, 
                      rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle, elevation_angle,
                      mesh_angle_r, mesh_angle_up, Obj_range, render_mode_type, diff_const, RCS_const):
   renderer = Renderer()

   render_mode = getattr(RenderMode, render_mode_type)
   
   # load mesh from outside
   script_directory = os.path.dirname(__file__)
   content_directory = os.path.join(script_directory, "../example-files/Avinash_RCS/Plate/")
   obj_filename = os.path.join(content_directory, "metal_plate_10cm_width.obj")#Plate_high_roughness_metal, Plate_high_roughness_flipped
   #print("start loading campus scene")
   load_Object(renderer, content_directory, obj_filename, mesh_angle_r, mesh_angle_up, diff_const)

   radiation_pattern_filename = os.path.join(script_directory, "../example-files/Radiation_pattern_new.txt")
   #radiation_pattern_filename = os.path.join(script_directory, "../example-files/test_radiation_pattern.txt")
   radiation_pattern, phi_axis, theta_axis = load_radiation_pattern_from_cst(radiation_pattern_filename)

   tx_antennas, rx_antennas = load_antennas_for_imaging_iwr6843AOP(From_vector, look_at_front, vec_up,
                                       radiation_pattern, phi_axis, theta_axis, Wavelength, rx_antenna_rad, azimuth_angle, elevation_angle)
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
   num_rays_rec = len(ray_channel_info.hit_positions)
   for mesh_id in ray_channel_info.mesh_ids:
      if [1,-1,-1] == ray_channel_info.mesh_ids:
         ray_multi += 1

   #iterate over mesh ids
      #if ray is direct -> x+1
      #else ray is multipath -> y+1
   
   # y+x = num_rays_rec
   print(f"received {num_rays_rec} rays")

   num_rays_sent = (image_width*image_height)*Oversamp_factor
   Power_ratio = (num_rays_rec/num_rays_sent)**2
   RCS_without_const = Power_ratio*(4*np.pi*(Obj_range**2))**2
   RCS_with_const = RCS_without_const * RCS_const
   
   azimuth_radi = np.deg2rad(azimuth_angle)
   elevation_radi = np.deg2rad(elevation_angle)

   # For the effective area, performing double integral over azimuth and elevation, which provides effective area in the omni directional sphere 
   ##########CORRECT VERSION:
   spheric_integral = lambda theta, phi : np.cos(phi)
   effective_area, error = dblquad(spheric_integral,
                                -elevation_radi, +elevation_radi, 
                                0, 2*azimuth_radi)
   #########################

   #############DEBUGGING####
   # spheric_integral = lambda theta, phi : np.sin(phi)
   # # spheric_integral = lambda theta, phi : np.cos(phi)
   # effective_area, error = dblquad(spheric_integral,
   #                              (np.pi/2-elevation_radi), np.pi/2,
   #                              0, 2*azimuth_radi)
   # effective_area *=2
   ############################
   # effective_area = 4*azimuth_radi*elevation_radi

   print(f"effective_area is: {effective_area:.4f}")
   # To calculate the gain factor
   gain_factor = (4 * np.pi) / effective_area
   print(f"gain_factor : {gain_factor:.4f}")
   gain_factor_dBsm = 20 * np.log10(gain_factor)
   ray_density_at_obj = num_rays_sent**2/gain_factor/(4*np.pi*(Obj_range**2))

   print(f"gain_factor_dBsm : {gain_factor_dBsm:.4f}")
   RCS_with_const_dBsm = 10 * np.log10(RCS_with_const) 
   normalized_RCS_dBsm = RCS_with_const_dBsm - gain_factor_dBsm

   print(f" RCS in dBsm {normalized_RCS_dBsm}") 

   return normalized_RCS_dBsm, gain_factor_dBsm, num_rays_rec, effective_area, RCS_with_const, ray_density_at_obj

def main():
    #Parameters for incident rays 
   image_width = 1200
   image_height = 600
   Oversamp_factor = 10
   Wavelength = 0.005

   # variables to change the range and theta
   Obj_range = 2
   iterations_azimuth = 18
   iterations_elevation = 18
   step_size_azimuth = 5 # in degree
   step_size_elevation = 5 # in degree
   rx_antenna_rad = 0.3

   look_at_front =  np.array([0.0, 0, 0.0])
   vec_up = np.array([0.0, 0.0, 1.0]) 
   
   render_mode_type = "RENDER_MODE_RAYTARGET_COMP"   #RENDER_MODE_RAYTARGET_COMP,  RENDER_MODE_GRAPHICS
   obj_width = 0.05
   obj_length = 0.05
   mesh_angle_r = 0
   mesh_angle_up = 90
   diff_const = 0.1
   # rx_antenna_rad = 0.0
   RCS_const = 0.3164 
   
   azimuth_angle_start =  calculate_azimuth_angle(obj_width, Obj_range)
   # azimuth_angle_start = 21.43
   elevation_angle_start =  calculate_elevation_angle(obj_length, Obj_range)

   normalized_rcs_over_angle = np.zeros((iterations_azimuth,iterations_elevation))
   gain_factor_over_angle = np.zeros((iterations_azimuth,iterations_elevation))
   number_rays_received_over_angle = np.zeros((iterations_azimuth,iterations_elevation))
   effective_ant_area_over_angle = np.zeros((iterations_azimuth,iterations_elevation))
   non_normalized_rcs_over_angle = np.zeros((iterations_azimuth,iterations_elevation))
   ray_density_at_obj_over_angle = np.zeros((iterations_azimuth,iterations_elevation))
   azimuth_angles = np.zeros((iterations_azimuth,1))
   elevation_angles = np.zeros((iterations_elevation,1))

   for az_iter_idx in range(iterations_azimuth):
      for elev_iter_idx in range(iterations_elevation):
         azimuth_angle = azimuth_angle_start + az_iter_idx*step_size_azimuth
         azimuth_angles[az_iter_idx] = azimuth_angle
         elevation_angle = elevation_angle_start + elev_iter_idx*step_size_elevation
         elevation_angles[elev_iter_idx] = elevation_angle
         # rx_antenna_rad += 0.01

         From_vector = np.array([Obj_range, 0.0, 0.0])
         print(f"The   azimuth_angle is {azimuth_angle:.2f}")
         print(f"The   elevation_angle is {elevation_angle:.2f}")
         print(f"The mesh_angle_up angle is set to {mesh_angle_up:.2f} degrees.")

         RCS_with_const_dBsm, gain_factor, number_rays_received, effective_ant_area, RCS_non_const_dBsm, ray_density_at_obj = simulate_plate(image_width, image_height, Oversamp_factor, Wavelength, 
                        rx_antenna_rad, From_vector, look_at_front, vec_up, azimuth_angle, elevation_angle,
                        mesh_angle_r, mesh_angle_up,  Obj_range, render_mode_type, diff_const, RCS_const)
         
         normalized_rcs_over_angle[az_iter_idx, elev_iter_idx] = RCS_with_const_dBsm
         gain_factor_over_angle[az_iter_idx, elev_iter_idx] = gain_factor
         number_rays_received_over_angle[az_iter_idx, elev_iter_idx] = number_rays_received
         effective_ant_area_over_angle[az_iter_idx, elev_iter_idx] = effective_ant_area
         non_normalized_rcs_over_angle[az_iter_idx, elev_iter_idx] = RCS_non_const_dBsm
         ray_density_at_obj_over_angle[az_iter_idx, elev_iter_idx] = ray_density_at_obj

   plt.plot(azimuth_angles, normalized_rcs_over_angle[:,0])
   plt.show()
   plt.figure()
   plt.plot(elevation_angles, normalized_rcs_over_angle[0,:])
   plt.show()
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   phi, theta = np.meshgrid(elevation_angles, azimuth_angles)
   surf = ax.plot_surface(phi, theta, normalized_rcs_over_angle, cmap='viridis')
   plt.show()
   debug_block = True
         
if __name__ =="__main__":
    main()