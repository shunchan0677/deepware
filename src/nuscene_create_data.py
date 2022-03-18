#!/usr/bin/python
# -*- coding: utf-8 -*-

# Load the SDK
import rospy
import math
from sensor_msgs.msg import Image as Image_ros
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from autoware_msgs.msg import DetectedObjectArray
from autoware_msgs.msg import DetectedObject
from geometry_msgs.msg import Quaternion
# from cv_bridge import CvBridge
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
#%matplotlib inline
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
from pyquaternion import Quaternion
from tqdm import tqdm


from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import transform_matrix, view_points, box_in_image, BoxVisibility
from nuscenes.utils.map_mask import MapMask


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
                               ax: Axes = None) -> None:
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

        sd_record = self.nusc.get('sample_data', sample_data_token)

        # Init axes.
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 9))

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
        mask_raster = map_mask._base_mask()

        print("before clop map")

        cropped = crop_image(mask_raster, pixel_coords[0], pixel_coords[1], int(scaled_limit_px * math.sqrt(2)))

        print("clop map")

        ypr_rad = Quaternion(pose['rotation']).yaw_pitch_roll
        yaw_deg = -math.degrees(ypr_rad[0])

        rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))
        ego_centric_map = crop_image(rotated_cropped, rotated_cropped.shape[1] / 2, rotated_cropped.shape[0] / 2,
                                     scaled_limit_px)
        #ax.imshow(ego_centric_map, extent=[-axes_limit, axes_limit, -axes_limit, axes_limit], cmap='gray', vmin=0,vmax=150)





if __name__=="__main__":
    rospy.init_node('nuscenes')
    #cv_bridge = CvBridge()

    pub_obj = rospy.Publisher(
            'detection/lidar_detector/objects', DetectedObjectArray, queue_size=1)
    pub_obj_mark = rospy.Publisher(
            'detection/lidar_detector/objects_mark', MarkerArray, queue_size=1)


    level5data = MyNuScenes(version='v1.01-train', dataroot='/media/brainiv/PioMeidai/v1.01-train', verbose=True)

    my_scene_token = level5data.scene[1]
    #level5data.render_scene_channel(my_scene_token["token"], 'CAM_FRONT',out_path='/media/brainiv/PioMeidai/sample.avi')

    sample = level5data.get('sample', level5data.scene[0]['first_sample_token'])
    camera_token = sample['data']['CAM_FRONT']
    cam = level5data.get('sample_data', camera_token)
    has_more_frames = True
    imsize = (640, 360)
    name = '{}: {} (Space to pause, ESC to exit)'.format(my_scene_token['name'], 'CAM_FRONT')

    lidar_token = sample['data']['LIDAR_TOP']
    lid = level5data.get('sample_data', lidar_token)
    level5data.explorer.render_ego_centric_map(lidar_token, 4, None)

    old_marker = MarkerArray()

    while has_more_frames:

        #lidar_sample = level5data.get('sample', cam['token'])
        lipath,boxes_l, ret = level5data.get_sample_data(lid['token'],
            box_vis_level=BoxVisibility.ANY)

        # Get data from DB
        impath, boxes, camera_intrinsic = level5data.get_sample_data(cam['token'],
            box_vis_level=BoxVisibility.ANY)

        # Load and render
        if not osp.exists(impath):
            raise Exception('Error: Missing image %s' % impath)
        im = cv2.imread(impath)
        # Render
        im = cv2.resize(im, imsize)
        #im_msg = cv_bridge.cv2_to_imgmsg(im, "bgr8")
        #pub.publish(im_msg)

        cv2.imshow(name, im)

        key = cv2.waitKey(100)  # Images stored at approx 10 Hz, so wait 10 ms.
        if key == 32:  # If space is pressed, pause.
            key = cv2.waitKey()

        if key == 27:  # if ESC is pressed, exit
            cv2.destroyAllWindows()
            break

        ego_pose = level5data.get("ego_pose",cam['ego_pose_token'])

        #print(boxes) # -- > Class list


        doa = DetectedObjectArray()
        doa_m = MarkerArray()
        doa.header.frame_id = "base_link"
        doa.header.stamp = rospy.Time.now()
        #doa_m.header.frame_id = "base_line"
        #doa_m.header.stamp = rospy.Time.now()

        bbox_list = []
        bbox_list_m = []
        i = 0
        for box in boxes_l:
            do = DetectedObject()
            do.id = i
            do.header.frame_id = "base_link"
            do.header.stamp = rospy.Time.now()
            corners = box.bottom_corners()
            do.pose.position.x = np.mean(corners[0,:])
            do.pose.position.y = np.mean(corners[1,:])
            do.pose.position.z = np.mean(corners[2,:])

            front_x = np.mean(corners[0,0:2]) - do.pose.position.x
            front_y = np.mean(corners[1,0:2]) - do.pose.position.y
            yaw = np.arctan(front_x/front_y)
            yaw_1 = np.cos(yaw/2.0)
            yaw_2 = 0.0
            yaw_3 = 0.0
            yaw_4 = np.sin(yaw/2.0)
            do.pose.orientation.x = yaw_1
            do.pose.orientation.y = yaw_2
            do.pose.orientation.z = yaw_3
            do.pose.orientation.w = yaw_4
            
            do.dimensions.x = front_x
            do.dimensions.y = front_y
            do.dimensions.z = 1.0
            
            bbox_list.append(do)


            marker_data = Marker()
            marker_data.id = i
            marker_data.header.frame_id = "base_link"
            marker_data.header.stamp = rospy.Time.now()
            marker_data.type = 1
            marker_data.lifetime = rospy.Duration()

            marker_data.ns = "basic_shapes"

            marker_data.action = Marker.ADD

            marker_data.color.r = 1.0
            marker_data.color.g = 0.0
            marker_data.color.b = 0.0
            marker_data.color.a = 1.0

            marker_data.scale.x = box.wlh[1]
            marker_data.scale.y = box.wlh[0]
            marker_data.scale.z = box.wlh[2]

            marker_data.pose.position.x = -box.center[0]
            marker_data.pose.position.y = -box.center[1]
            marker_data.pose.position.z = 0.0
            marker_data.pose.orientation.x = box.orientation[1]
            marker_data.pose.orientation.y = box.orientation[2]
            marker_data.pose.orientation.z = box.orientation[3]
            marker_data.pose.orientation.w = box.orientation[0]

            bbox_list_m.append(marker_data)
            i+=1


        if(len(old_marker.markers)>i):
             for k in range(len(old_marker.markers)-i):
                  marker_data = Marker()
                  marker_data.id = i + k
                  marker_data.header.frame_id = "base_link"
                  marker_data.header.stamp = rospy.Time.now()
                  marker_data.type = 1
                  marker_data.lifetime = rospy.Duration()

                  marker_data.ns = "basic_shapes"
                  marker_data.action = Marker.DELETE
                  bbox_list_m.append(marker_data)
                  

        doa.objects = bbox_list
        pub_obj.publish(doa)

        doa_m.markers = bbox_list_m
        pub_obj_mark.publish(doa_m)

        old_marker = doa_m


        if not cam['next'] == "":
            cam = level5data.get('sample_data', cam['next'])
            lid = level5data.get('sample_data', lid['next'])
        else:
            has_more_frames = False





