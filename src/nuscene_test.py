#!/usr/bin/python
# -*- coding: utf-8 -*-

# Load the SDK
#import rospy
import math
#from sensor_msgs.msg import Image as Image_ros
#from visualization_msgs.msg import Marker
#from visualization_msgs.msg import MarkerArray
#from autoware_msgs.msg import DetectedObjectArray
#from autoware_msgs.msg import DetectedObject
#from geometry_msgs.msg import Quaternion
# from cv_bridge import CvBridge
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import matplotlib.pyplot
matplotlib.pyplot.switch_backend('agg')
import os
import os.path as osp
from nuscenes.nuscenes import NuScenes
from nuscenes.nuscenes import NuScenesExplorer
from typing import Tuple, List
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from PIL import Image
from matplotlib.axes import Axes
import matplotlib.patches as pat
from pyquaternion import Quaternion
from tqdm import tqdm
from functools import reduce

from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import transform_matrix, view_points, box_in_image, BoxVisibility
from nuscenes.utils.map_mask import MapMask
import csv
#plt.style.use('dark_background')

# Load the dataset
# Adjust the dataroot parameter below to point to your local dataset path.
# Note that using "~" for your home directory typically won't work here, thus specify the complete pathname.
# The correct dataset path contains at least the following four folders (or similar): images, lidar, maps, v1.0-mini
# In case you didn't download the 'v1.0-mini' version of the dataset, also adjust the version parameter below.
class MyNuScenes(NuScenes):
    def __init__(self,
                 version: str = 'v1.0-mini',
                 dataroot: str = '/data/sets/nuscenes',
                 verbose: bool = True,
                 map_resolution: float = 0.1):
        super().__init__(
                 version,
                 dataroot,
                 verbose,
                 map_resolution)
        self.explorer = MyNuScenesExplorer(self)

class MyNuScenesExplorer(NuScenesExplorer):
    def __init__(self, nusc: MyNuScenes):
        super().__init__(nusc)
        self.nusc = nusc

    def render_scene_channel(self,
                             scene_token: str,
                             channel: str = 'CAM_FRONT',
                             freq: float = 10,
                             imsize: Tuple[float, float] = (640, 360),
                             out_path: str = None) -> None:
        """
        Renders a full scene for a particular camera channel.
        :param scene_token: Unique identifier of scene to render.
        :param channel: Channel to render.
        :param freq: Display frequency (Hz).
        :param imsize: Size of image to render. The larger the slower this will run.
        :param out_path: Optional path to write a video file of the rendered frames.
        """

        valid_channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                          'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

        assert imsize[0] / imsize[1] == 16 / 9, "Aspect ratio should be 16/9."
        assert channel in valid_channels, 'Input channel {} not valid.'.format(channel)

        if out_path is not None:
            assert osp.splitext(out_path)[-1] == '.avi'

        # Get records from DB
        scene_rec = self.nusc.get('scene', scene_token)
        sample_rec = self.nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = self.nusc.get('sample_data', sample_rec['data'][channel])

        # Open CV init
        name = '{}: {} (Space to pause, ESC to exit)'.format(scene_rec['name'], channel)
        cv2.namedWindow(name)
        cv2.moveWindow(name, 0, 0)

        if out_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(out_path, fourcc, freq, imsize)
        else:
            out = None

        has_more_frames = True
        while has_more_frames:

            # Get data from DB
            impath, boxes, camera_intrinsic = self.nusc.get_sample_data(sd_rec['token'],
                                                                        box_vis_level=BoxVisibility.ANY)

            # Load and render
            if not osp.exists(impath):
                raise Exception('Error: Missing image %s' % impath)
            im = cv2.imread(impath)
            #for box in boxes:
            #    c = self.get_color(box.name)
            #    box.render_cv2(im, view=camera_intrinsic, normalize=True, colors=(c, c, c))

            # Render
            im = cv2.resize(im, imsize)
            cv2.imshow(name, im)
            if out_path is not None:
                out.write(im)

            key = cv2.waitKey(10)  # Images stored at approx 10 Hz, so wait 10 ms.
            if key == 32:  # If space is pressed, pause.
                key = cv2.waitKey()

            if key == 27:  # if ESC is pressed, exit
                cv2.destroyAllWindows()
                break

            if not sd_rec['next'] == "":
                sd_rec = self.nusc.get('sample_data', sd_rec['next'])
            else:
                has_more_frames = False

        cv2.destroyAllWindows()
        if out_path is not None:
            out.release()

    def render_ego_centric_map(self,
                               sample_data_token: str,
                               axes_limit: float = 40,
                               ax: Axes = None,
                               k: int = 0,
                               j: int = 0)-> None:
        """
        Render map centered around the associated ego pose.
        :param sample_data_token: Sample_data token.
        :param axes_limit: Axes limit measured in meters.
        :param ax: Axes onto which to render.
        """

        def crop_image(image: np.array,
                       x_px: int,
                       y_px: int,
                       axes_limit_px: int) -> np.array:
            x_min = int(x_px - axes_limit_px)
            x_max = int(x_px + axes_limit_px)
            y_min = int(y_px - axes_limit_px)
            y_max = int(y_px + axes_limit_px)

            cropped_image = image[y_min:y_max, x_min:x_max]

            return cropped_image

        #sd_record = self.nusc.get('sample_data', sample_data_token)
        sd_record = sample_data_token

        # Init axes.
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(4, 4))

        print ("set axis")
        sample = self.nusc.get('sample', sd_record['sample_token'])
        scene = self.nusc.get('scene', sample['scene_token'])
        log = self.nusc.get('log', scene['log_token'])
        map = self.nusc.get('map', log['map_token'])
        map_mask = map['mask']

        print("read map")

        pose = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        pixel_coords = map_mask.to_pixel_coords(pose['translation'][0], pose['translation'][1])

        print("read map mask")

        scaled_limit_px = int(axes_limit * (1.0 / map_mask.resolution))
        mask_raster = map_mask.mask()

        print("before clop map")

        cropped = crop_image(mask_raster, pixel_coords[0], pixel_coords[1], int(scaled_limit_px * 2))#math.sqrt(2)))

        print("clop map")

        ypr_rad = Quaternion(pose['rotation']).yaw_pitch_roll
        yaw_deg = 90.0-math.degrees(ypr_rad[0])

        rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))
        print(rotated_cropped.shape)
        ego_centric_map = crop_image(rotated_cropped, rotated_cropped.shape[1] / 2 , rotated_cropped.shape[0] / 2,# - scaled_limit_px*0.3,
                                     scaled_limit_px)
        print(ego_centric_map.shape)
        ax.imshow(ego_centric_map, extent=[-axes_limit, axes_limit, -1.0*axes_limit, 1.0*axes_limit], cmap='gray', vmin=0,vmax=150)
        margin = 40.0
        ax.set_xlim([ - margin, margin])
        ax.set_ylim([ - 0.4*margin+1 , 1.6*margin+1])
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        os.makedirs("/home/s_seiya/workspace5/level5dataset/v1.01-train/dataset/scene"+str(k).zfill(5)+"/data"+str(j).zfill(5), exist_ok=True)
        plt.savefig("/home/s_seiya/workspace5/level5dataset/v1.01-train/dataset/scene"+str(k).zfill(5)+"/data"+str(j).zfill(5)+"/map.png")
        plt.clf()


if __name__=="__main__":

    level5data = MyNuScenes(version='v1.01-train', dataroot='/home/s_seiya/workspace2/level5dataset/v1.01-train', verbose=True)

    #my_scene_token = level5data.scene[1]
    #level5data.render_scene_channel(my_scene_token["token"], 'CAM_FRONT',out_path='/media/brainiv/PioMeidai/sample.avi')

    for k in range(len(level5data.scene)):
      if(k > 144):
       my_scene_token = level5data.scene[k]
       sample = level5data.get('sample', level5data.scene[k]['first_sample_token'])
       camera_token = sample['data']['CAM_FRONT']
       cam = level5data.get('sample_data', camera_token)
       has_more_frames = True
       imsize = (640, 360)
       name = '{}: {} (Space to pause, ESC to exit)'.format(my_scene_token['name'], 'CAM_FRONT')

       lidar_token = sample['data']['LIDAR_TOP']
       lid = level5data.get('sample_data', lidar_token)
       lid_f = lid

       for l in range(5):
           cam = level5data.get('sample_data', cam['next'])
           lid = level5data.get('sample_data', lid['next'])
       j = 0
       while has_more_frames:
          plt.clf()
          #rendering map
          level5data.explorer.render_ego_centric_map(lid, 80.0, None,k,j)
          #rendering image
          impath, boxes, camera_intrinsic = level5data.get_sample_data(cam['token'],
            box_vis_level=BoxVisibility.ANY)
          im = cv2.imread(impath)
          cv2.imwrite("/home/s_seiya/workspace5/level5dataset/v1.01-train/dataset/scene"+str(k).zfill(5)+"/data"+str(j).zfill(5)+"/image.png",im)

          #rendering lidar
          for l in range(5):
              if(l == 0):
                  old_lid = level5data.get('sample_data', lid['prev'])
              else:
                  old_lid = level5data.get('sample_data', old_lid['prev'])
          for l in range(10):
              if (l == 0):
                      lid2 = level5data.get('sample_data', old_lid['next'])
              else:
                  if not lid2["next"] == "":
                      lid2 = level5data.get('sample_data', lid2['next'])
                  else:
                      has_more_frames = False
                      break

              pose = level5data.get('ego_pose', lid['ego_pose_token'])
              car_from_global = transform_matrix(pose['translation'],
                                                                   Quaternion(pose['rotation']), inverse=True)

              calib = level5data.get('calibrated_sensor', lid['calibrated_sensor_token'])
              ref_from_car = transform_matrix(calib['translation'],
                                 Quaternion(calib['rotation']), inverse=True)

              pose_c = level5data.get('ego_pose', lid2['ego_pose_token'])
              global_from_car = transform_matrix(pose_c['translation'],
                                    Quaternion(pose_c['rotation']), inverse=False)
              calib_c = level5data.get('calibrated_sensor', lid2['calibrated_sensor_token'])
              car_from_current = transform_matrix(calib_c['translation'],
                                            Quaternion(calib_c['rotation']), inverse=False)
              #ypr_rad_c = Quaternion(pose['rotation']).yaw_pitch_roll[0]
             
              yaw_deg = math.radians(-90.0)

              data_path, boxes, camera_intrinsic = level5data.get_sample_data(lid2['token'])
              fig, axes = plt.subplots(1, 1, figsize=(4, 4))
              trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
              view_mat = np.array([[np.cos(yaw_deg),-np.sin(yaw_deg),0,0],[np.sin(yaw_deg),np.cos(yaw_deg),0,0],[0,0,1,0],[0,0,0,1]])
              pc = LidarPointCloud.from_file(data_path)
              pc.transform(trans_matrix)
              pc.render_height(axes, view=view_mat)

              margin = 40.0
              axes.set_xlim([ - margin, margin])
              axes.set_ylim([ - 0.4*margin , 1.6*margin])
              plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
              plt.gca().spines['left'].set_visible(False)
              plt.gca().spines['top'].set_visible(False)
              plt.savefig("/home/s_seiya/workspace5/level5dataset/v1.01-train/dataset/scene"+str(k).zfill(5)+"/data"+str(j).zfill(5)+"/lidar_"+str(l)+'.png')
              plt.clf()

              fig, axes = plt.subplots(1, 1, figsize=(4, 4))
              #object render
              for box in boxes:
                  c = np.array(level5data.explorer.get_color(box.name)) / 255.0
                  #rotate_box = box.transfrom(trans_matrix)
                  #self.points[:3, :] = transf_matrix.dot(np.vstack((self.points[:3, :], np.ones(self.nbr_points()))))[:3, :]
                  trans_matrix_ = reduce(np.dot, [view_mat, ref_from_car, car_from_global, global_from_car, car_from_current])
                  points = trans_matrix_.dot(np.vstack((box.bottom_corners()[:3, :], np.ones(4))))[:3, :]
                  #box.center = transf_matrix.dot(np.vstack((box.center[:3, :], np.ones(1))))[:3, :]
                  #box.render(axes, view=view_mat, colors=(c, c, c))
                  poly = plt.Polygon((points[0:2,0],points[0:2,1],points[0:2,2],points[0:2,3]),color=(0,1.0,0))
                  axes.add_patch(poly)

              axes.set_xlim([ - margin, margin])
              axes.set_ylim([ - 0.4*margin , 1.6*margin])
              plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
              plt.gca().spines['left'].set_visible(False)
              plt.gca().spines['top'].set_visible(False)
              plt.savefig("/home/s_seiya/workspace5/level5dataset/v1.01-train/dataset/scene"+str(k).zfill(5)+"/data"+str(j).zfill(5)+"/bbox_"+str(l)+'.png')
              plt.clf()

          #old ego pose
          fig, axes2 = plt.subplots(1, 1, figsize=(4, 4))
          for order in range(5):
                  if(order == 0):
                      old_lid = level5data.get('sample_data', lid['prev'])
                  else:
                      old_lid = level5data.get('sample_data', old_lid['prev'])
          for p in range(5):
                   if(order == 0):
                      old_lid = old_lid
                   else:
                       old_lid = level5data.get('sample_data', old_lid['next'])
                   ego_vehicle_x = 1.695/2.0
                   ego_vehicle_y = 4.36/2.0
                   pose = level5data.get('ego_pose', lid['ego_pose_token'])
                   car_from_global = transform_matrix(pose['translation'],Quaternion(pose['rotation']), inverse=True)
                   calib = level5data.get('calibrated_sensor', lid['calibrated_sensor_token'])
                   ref_from_car = transform_matrix(calib['translation'],Quaternion(calib['rotation']), inverse=True)

                   pose_c = level5data.get('ego_pose', old_lid['ego_pose_token'])
                   global_from_car = transform_matrix(pose_c['translation'],Quaternion(pose_c['rotation']), inverse=False)
                   calib_c = level5data.get('calibrated_sensor', old_lid['calibrated_sensor_token'])
                   car_from_current = transform_matrix(calib_c['translation'],Quaternion(calib_c['rotation']), inverse=False)

                   points = np.array([[-ego_vehicle_y,-ego_vehicle_x],[-ego_vehicle_y,ego_vehicle_x],[ego_vehicle_y,ego_vehicle_x],[ego_vehicle_y,-ego_vehicle_x]])
                   trans_matrix2 = reduce(np.dot, [view_mat,ref_from_car, car_from_global, global_from_car, car_from_current])
                   points = trans_matrix2.dot(np.vstack((points.T, np.ones([2,4]))))[:2, :]
                   poly = plt.Polygon((points[0:2,0],points[0:2,1],points[0:2,2],points[0:2,3]),color=(0.5*(p+1)/5.0,0,0))
                   axes2.add_patch(poly)
          ans_list = []
          for num_step in range(5):
                   old_lid = level5data.get('sample_data', old_lid['next'])
                   pose_c = level5data.get('ego_pose', old_lid['ego_pose_token'])
                   vec_x = pose_c['translation'][0] - pose['translation'][0]
                   vec_y = pose_c['translation'][1] - pose['translation'][1]
                   yaw = Quaternion(pose['rotation']).yaw_pitch_roll[0]
                   ans_x = (vec_x*np.cos(-yaw) - vec_y*np.sin(-yaw))
                   ans_y = (vec_x*np.sin(-yaw) + vec_y*np.cos(-yaw))
                   yaw_diff = Quaternion(pose_c['rotation']).yaw_pitch_roll[0] - yaw
                   ans_list += [ans_x,ans_y,yaw_diff]
          with open("/home/s_seiya/workspace5/level5dataset/v1.01-train/dataset/scene"+str(k).zfill(5)+".csv", 'a') as csvfile:
                   writer = csv.writer(csvfile, lineterminator='\n')
                   writer.writerow(ans_list)
          axes2.set_xlim([ - margin, margin])
          axes2.set_ylim([ - 0.4*margin , 1.6*margin])
          plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
          plt.gca().spines['left'].set_visible(False)
          plt.gca().spines['top'].set_visible(False)
          plt.savefig("/home/s_seiya/workspace5/level5dataset/v1.01-train/dataset/scene"+str(k).zfill(5)+"/data"+str(j).zfill(5)+"/ego_pose.png")
          plt.clf()
          wp_flag = True
          fig, axes2 = plt.subplots(1, 1, figsize=(4, 4))
          old_lid = lid_f
          ego_vehicle_x = 1.695#*2.0
          ego_vehicle_y = 4.36#*2.0
          num = 0
          while(wp_flag):
              pose = level5data.get('ego_pose', lid['ego_pose_token'])
              car_from_global = transform_matrix(pose['translation'],Quaternion(pose['rotation']), inverse=True)
              calib = level5data.get('calibrated_sensor', lid['calibrated_sensor_token'])
              ref_from_car = transform_matrix(calib['translation'],Quaternion(calib['rotation']), inverse=True)

              pose_c = level5data.get('ego_pose', old_lid['ego_pose_token'])
              global_from_car = transform_matrix(pose_c['translation'],Quaternion(pose_c['rotation']), inverse=False)
              calib_c = level5data.get('calibrated_sensor', old_lid['calibrated_sensor_token'])
              car_from_current = transform_matrix(calib_c['translation'],Quaternion(calib_c['rotation']), inverse=False)

              points = np.array([[-ego_vehicle_y,-ego_vehicle_x],[-ego_vehicle_y,ego_vehicle_x],[ego_vehicle_y,ego_vehicle_x],[ego_vehicle_y,-ego_vehicle_x]])
              trans_matrix2 = reduce(np.dot, [view_mat, ref_from_car, car_from_global, global_from_car, car_from_current])
              points = trans_matrix2.dot(np.vstack((points.T, np.ones([2,4]))))[:2, :]
              poly = plt.Polygon((points[0:2,0],points[0:2,1],points[0:2,2],points[0:2,3]),color=(0,0,1.0))
              axes2.add_patch(poly)
              num += 1
              if not old_lid["next"] == "":
                  old_lid = level5data.get('sample_data', old_lid['next'])
              else:
                  wp_flag = False

          axes2.set_xlim([ - margin, margin])
          axes2.set_ylim([ - 0.4*margin , 1.6*margin])
          plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
          plt.gca().spines['left'].set_visible(False)
          plt.gca().spines['top'].set_visible(False)
          plt.savefig("/home/s_seiya/workspace5/level5dataset/v1.01-train/dataset/scene"+str(k).zfill(5)+"/data"+str(j).zfill(5)+"/waypoint.png")
          plt.clf()
          
          next_lid = lid
          for next_ in range(5):
              next_lid = level5data.get('sample_data', next_lid['next'])

          if not next_lid['next'] == "":
            cam = level5data.get('sample_data', cam['next'])
            lid = level5data.get('sample_data', lid['next'])
          else:
            has_more_frames = False
          j+=1

