#!/usr/bin/python
# -*- coding: utf-8 -*-
import rospy
import numpy as np
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
import tf
import cv2
from sensor_msgs.msg import Image

from std_msgs.msg import Bool
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointCloud
from autoware_msgs.msg import DetectedObjectArray
from autoware_msgs.msg import DetectedObject
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
import argparse
from geometry_msgs.msg import Point32
import ros_numpy
import pcl
import time
# . install/local_setup.bash


class FeatureExtractor():
    def __init__(self,flag):
        self.arg_devide_flag = flag
        self.objects_list = []
        self.vm_flag = True
        self.marker = MarkerArray()
        self.waypoint = MarkerArray()
        self.tlr = Bool()
        self.tlr.data = False
        self.tlr_count = 0
        self.map_points = PointCloud()
        self.no_ground_points = PointCloud()
        self.bridge = CvBridge()
        self.listener = tf.TransformListener()
        self.sub_vec = rospy.Subscriber(
            'vector_map_carla', MarkerArray, self.callback_vec)
        self.sub_pc = rospy.Subscriber(
            'points_raw', PointCloud2, self.callback_pc)
        self.sub_obj = rospy.Subscriber(
            'object_markers', MarkerArray, self.callback_obj)
        self.sub_tlr = rospy.Subscriber(
            'light_color', Bool, self.callback_tlr)
        self.sub_wp = rospy.Subscriber(
            'global_waypoints_mark', MarkerArray, self.callback_wp)
        self.pub_ego = rospy.Publisher(
            'ego_vehicle_marker', Marker, queue_size=10)
        self.image_pub = rospy.Publisher(
            "vector_image_raw", Image, queue_size=10)
        if(self.arg_devide_flag):
            self.image_wp_pub = rospy.Publisher(
                "vector_image_raw/waypoint", Image, queue_size=10)
            self.image_hd_pub = rospy.Publisher(
                "vector_image_raw/hd_map", Image, queue_size=10)
            self.image_obj_pub = rospy.Publisher(
                "vector_image_raw/objects", Image, queue_size=10)
            self.image_ego_pub = rospy.Publisher(
                "vector_image_raw/ego_vehicle", Image, queue_size=10)
            self.image_without_ego_pub = rospy.Publisher(
                "vector_image_raw/without_ego_vehicle", Image, queue_size=10)
            self.image_points_pub = rospy.Publisher(
                "vector_image_raw/points", Image, queue_size=10)
        """
        self.pc_pub_list = []
        for i in range(10):
            self.pc_pub_list.append(rospy.Publisher(
                "points_list_"+str(i), PointCloud, queue_size=10))
        """
        self.ego_vehicle_x = 4.36
        self.ego_vehicle_y = 1.695
        self.ego_vehicle_z = 1.46
        self.old_tf = []
        self.old_pc = []
        self.world_frame = "map"
        self.ego_frame = "base_link"
        self.height = 400
        self.width = 400
        self.rate = 60.0
        self.center_point_x = int(4 * self.width / 5.0)
        self.center_point_y = self.height / 2
        self.polygon_flag = False
        self.vmap_tf = PoseStamped()
        self.color = (255, 0, 0)
        self.br = tf.TransformBroadcaster()
        self.trans = (0,0,0)
        self.rot = (0,0,0,1)

    def set_bbox(self, obj, color_map, image_np):
        # put bbox on image
        box_pose = PoseStamped()
        box_pose.header.frame_id = obj.header.frame_id
        box_pose.header.stamp = rospy.Time.now() - rospy.Duration((12-5)*0.1)
        box_pose.pose = obj.pose
        box_pose_velo = PoseStamped()
        try:
            box_pose_velo = self.listener.transformPose(self.ego_frame, box_pose)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print"error"
        box_ori = box_pose_velo.pose.orientation
        box_yaw = - tf.transformations.euler_from_quaternion(
            (box_ori.x, box_ori.y, box_ori.z, box_ori.w))[2]

        image_center_point = np.array(
            [self.center_point_x, self.center_point_y])
        box_position_velo = np.array(
            [box_pose_velo.pose.position.x, box_pose_velo.pose.position.y])

        rot_mat = np.array(
            [[np.cos(box_yaw), -np.sin(box_yaw)], [np.sin(box_yaw), np.cos(box_yaw)]])
        box_size = np.array([obj.dimensions.x / 2.0, obj.dimensions.y / 2.0])
        box_mat = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])
        box_4 = box_size * box_mat
        rot_res = np.dot(box_4, rot_mat)
        rot_res_rated = (box_position_velo + rot_res) * self.height / self.rate
        image_box = image_center_point.astype(
            np.int32) - rot_res_rated.astype(np.int32)

        contours = image_box[:, [1, 0]]
        cv2.fillConvexPoly(image_np, points=contours,  color=color_map)

        return image_np

    def pub_ego_bbox(self):
        # put ego vehicle bbox in rviz
        marker_data = Marker()
        marker_data.header.frame_id = self.world_frame
        marker_data.header.stamp = rospy.Time.now()
        marker_data.type = 1

        marker_data.ns = "basic_shapes"

        marker_data.action = Marker.ADD

        marker_data.color.r = 1.0
        marker_data.color.g = 0.0
        marker_data.color.b = 0.0
        marker_data.color.a = 1.0

        marker_data.scale.x = self.ego_vehicle_x
        marker_data.scale.y = self.ego_vehicle_y
        marker_data.scale.z = self.ego_vehicle_z

        marker_data.pose.position.x = self.trans[0]
        marker_data.pose.position.y = self.trans[1]
        marker_data.pose.position.z = self.trans[2]
        marker_data.pose.orientation.x = self.rot[0]
        marker_data.pose.orientation.y = self.rot[1]
        marker_data.pose.orientation.z = self.rot[2]
        marker_data.pose.orientation.w = self.rot[3]
        self.pub_ego.publish(marker_data)

    def update_map_points(self):
        ego_pose = PoseStamped()
        ego_pose.header.frame_id = self.world_frame
        ego_pose.header.stamp = rospy.Time.now() - rospy.Duration((12-5)*0.1)
        ego_pose.pose.position.x = self.trans[0]
        ego_pose.pose.position.y = self.trans[1]
        ego_pose.pose.position.z = self.trans[2]
        ego_pose.pose.orientation.x = self.rot[0]
        ego_pose.pose.orientation.y = self.rot[0]
        ego_pose.pose.orientation.z = self.rot[0]
        ego_pose.pose.orientation.w = self.rot[0]
        ego_pose_map = self.listener.transformPose("map", ego_pose)
        self.map_points.points = []
        for mark in self.marker.markers:
            if(True):
                for i in range(len(mark.points) / 2):
                    mean_x = (mark.points[2 * i].x +
                              mark.points[2 * i + 1].x) / 2.0
                    mean_y = (mark.points[2 * i].y +
                              mark.points[2 * i + 1].y) / 2.0
                    dis = np.sqrt((mean_x - ego_pose_map.pose.position.x)
                                  ** 2 + (mean_y - ego_pose_map.pose.position.y)**2)
                    if(dis < self.rate * 2.0):
                        self.map_points.points.append(mark.points[2 * i])
                        self.map_points.points.append(mark.points[2 * i + 1])


    def update_image_points(self, image_np):
        for p in pc2.read_points(self.no_ground_points, field_names = ("x", "y", "z"), skip_nans=True):
           x = int(self.center_point_x - p[0] * self.height / self.rate)
           y = int(self.center_point_y - p[1] * self.height / self.rate)
           if(abs(x) < self.height and abs(y) < self.height):
              image_np[x,y]=(255,255,255)
        return image_np


    def update_image_waypoint(self, image_np):
        # Update map for waypoints
        way_points = PointCloud()
        way_points.header.frame_id = "map"
        way_points.header.stamp = rospy.Time.now() - rospy.Duration((12-5)*0.1)
        for waypoint in self.waypoint.markers:
            if(waypoint.ns == "global_change_flag_lane_1"):
                point = Point()
                point = waypoint.pose.position
                way_points.points.append(point)
      
        rotate_points = PointCloud()
        try:
            rotate_points = self.listener.transformPointCloud(
                self.ego_frame, way_points)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print"error"

        if (self.tlr.data == True):
            self.color = (0, 0, 255)
            self.tlr_count = 1
        elif(self.tlr.data == False):
            self.tlr_count += 1
            if(self.tlr_count > 30):
                self.color = (255, 0, 0)
                self.tlr_count = 1

        for i in range(len(rotate_points.points) - 1):
            cv2.line(image_np, (self.center_point_y - int(rotate_points.points[i].y * self.height / self.rate), self.center_point_x - int(rotate_points.points[i].x * self.height / self.rate)), (
                self.center_point_y - int(rotate_points.points[i + 1].y * self.height / self.rate), self.center_point_x - int(rotate_points.points[i + 1].x * self.height / self.rate)), self.color, 10)
        return image_np

    def update_image_HDmap(self, image_np):
        # Update map for HD Map
        self.map_points.header.frame_id = "map"
        self.map_points.header.stamp = rospy.Time.now() - rospy.Duration((12-5)*0.1)
        dis = np.sqrt((self.trans[0] - self.vmap_tf.pose.position.x)
                      ** 2 + (self.trans[1] - self.vmap_tf.pose.position.y)**2)
        if(dis > self.rate * 1.0):
            self.vmap_tf.pose.position.x = self.trans[0]
            self.vmap_tf.pose.position.y = self.trans[1]
            self.update_map_points()

      
        rotate_points = PointCloud()
        try:
            rotate_points = self.listener.transformPointCloud(
                self.ego_frame, self.map_points)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print"error"

        for i in range(len(rotate_points.points) / 2):
            cv2.line(image_np, (self.center_point_y - int(rotate_points.points[2 * i].y * self.height / self.rate), self.center_point_x - int(rotate_points.points[2 * i].x * self.height / self.rate)), (
                self.center_point_y - int(rotate_points.points[2 * i + 1].y * self.height / self.rate), self.center_point_x - int(rotate_points.points[2 * i + 1].x * self.height / self.rate)), (255, 255, 255),2)
        return image_np

    def update_image_obj(self, image_np):
        # Update map for Historical Detected Objects
        objects_list = self.objects_list
        for i, objects in enumerate(objects_list):
          if(i < 5):
            for obj in objects.objects:
                if(False):
                    # for polygon
                    point_list = []
                    for point in obj.convex_hull.polygon.points:
                        point_list.append(
                            [center_point_y - int(point.y * height / rate), center_point_x - int(point.x * height / rate)])
                    contours = np.array(point_list)
                    cv2.fillConvexPoly(image_np, points=contours, color=(
                        0, int(128.0 + 128.0 * (i) / len(objects_list) - 1), 0))
                else:
                    # for 3d bbox
                    color_map = (0, int(128.0 + 128.0 * (i) /
                                        len(objects_list) - 1), 0)
                    image_np = self.set_bbox(obj, color_map, image_np)
        return image_np

    def update_image_ego(self, image_np):
        # Update map for Historical Ego States
        for i in range(len(self.old_tf)):
            old_tf = self.old_tf[i]
            ego = DetectedObject()
            ego.header.frame_id = self.world_frame
            ego.pose.position.x = old_tf[0][0]
            ego.pose.position.y = old_tf[0][1]
            ego.pose.position.z = old_tf[0][2]
            ego.pose.orientation.x = old_tf[1][0]
            ego.pose.orientation.y = old_tf[1][1]
            ego.pose.orientation.z = old_tf[1][2]
            ego.pose.orientation.w = old_tf[1][3]
            ego.dimensions.x = self.ego_vehicle_x
            ego.dimensions.y = self.ego_vehicle_y
            ego.dimensions.z = self.ego_vehicle_z
            color_map = (0, 0, int(128.0 + 128.0 * (i) / len(self.old_tf) - 1))
            image_np = self.set_bbox(ego, color_map, image_np)
        return image_np


    def add_image(self, image_1, image_2):
        img2gray = cv2.cvtColor(image_2,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_bg = cv2.bitwise_and(image_1,image_1,mask = mask_inv)
        img2_fg = cv2.bitwise_and(image_2,image_2,mask = mask)
        result = cv2.add(img1_bg,img2_fg)
        return result

    def callback_pc(self, points_clound):
        self.no_ground_points = points_clound

        #self.listener.waitForTransform(self.world_frame, self.ego_frame, rospy.Time.now(), rospy.Duration(0.5))
        try:
             self.listener.waitForTransform(self.world_frame, self.ego_frame, rospy.Time.now(), rospy.Duration(0.5))
             (self.trans, self.rot) = self.listener.lookupTransformFull("/map", rospy.Time.now(), "/base_link", rospy.Time.now()-rospy.Duration(0.1),self.world_frame)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print"error"
        #(self.trans, self.rot) = self.listener.lookupTransform(self.world_frame, self.ego_frame, rospy.Time.now())
        self.pub_ego_bbox()
        self.old_tf.append([self.trans, self.rot])


        """
        data = pc2.read_points(points_clound, skip_nans=True, field_names=("x", "y", "z"))
        pc = PointCloud()
        pc.header.frame_id = "velodyne"
        for p in data:
            if(p[0]**2 + p[1]**2 < 8100):
                points = Point32()
                points.x = p[0]
                points.y = p[1]
                points.z = p[2]
                pc.poinrvizts.append(points)
        pc = self.listener.transformPointCloud("map", pc)

        self.old_pc.append(pc)
        """

        if(len(self.old_tf) > 10):
            self.old_tf.pop(0)
            #self.old_pc.pop(0)


        if(len(self.old_tf) == 10):
            # publish old tf
            """
            for i in range(10):
                (trans, rot) = self.listener.lookupTransformFull("/world", rospy.Time.now(), "/base_link", rospy.Time.now() - rospy.Duration((11-i)*0.1),self.world_frame)
                self.br.sendTransform(trans, rot,rospy.Time.now(),"old_tf_"+str(i),self.world_frame)
            """
            # publish point_cloud_list
            #for i in range(len(self.old_tf)):
                #pc = self.old_pc[i]
                #pc = self.listener.transformPointCloud("old_tf_"+str(i), pc)
                #self.pc_pub_list[i].publish(pc)
                

            # create maps
            image_np_way = np.zeros((self.height, self.width, 3), np.uint8)
            image_np_hd = np.zeros((self.height, self.width, 3), np.uint8)
            image_np_obj = np.zeros((self.height, self.width, 3), np.uint8)
            image_np_ego = np.zeros((self.height, self.width, 3), np.uint8)
            image_np_points = np.zeros((self.height, self.width, 3), np.uint8)

            size_in_car_x = int(self.ego_vehicle_x * self.height / self.rate)
            size_in_car_y = int(self.ego_vehicle_y * self.width / self.rate)

            # update map
            image_np_way = self.update_image_waypoint(image_np_way)
            image_np_hd = self.update_image_HDmap(image_np_hd)
            image_np_obj = self.update_image_obj(image_np_obj)
            image_np_ego = self.update_image_ego(image_np_ego)
            #image_np_points = self.update_image_points(image_np_points)

            image_np_tmp = self.add_image(image_np_way,image_np_hd)
            image_np_without_ego = self.add_image(image_np_tmp,image_np_obj)
            image_np_all = self.add_image(image_np_without_ego,image_np_ego)

            if(self.arg_devide_flag):
                msg = self.bridge.cv2_to_imgmsg(image_np_way, encoding="bgr8")
                self.image_wp_pub.publish(msg)
                msg = self.bridge.cv2_to_imgmsg(image_np_points, encoding="bgr8")
                self.image_points_pub.publish(msg)
                msg = self.bridge.cv2_to_imgmsg(image_np_hd, encoding="bgr8")
                self.image_hd_pub.publish(msg)
                msg = self.bridge.cv2_to_imgmsg(image_np_obj, encoding="bgr8")
                self.image_obj_pub.publish(msg)
                msg = self.bridge.cv2_to_imgmsg(image_np_ego, encoding="bgr8")
                self.image_ego_pub.publish(msg)
                msg = self.bridge.cv2_to_imgmsg(image_np_without_ego, encoding="bgr8")
                self.image_without_ego_pub.publish(msg)

            # Publish map
            msg = self.bridge.cv2_to_imgmsg(image_np_all, encoding="bgr8")
            self.image_pub.publish(msg)

    def callback_vec(self, MarkerArray):
        if(self.vm_flag):
            self.vm_flag = False
            self.marker = MarkerArray

    def callback_obj(self, Objects):
        converted_obj_array = DetectedObjectArray()
        converted_obj_list = []
        for obj in Objects.markers:
            converted_obj = DetectedObject()
            box_pose = PoseStamped()
            box_pose.header.frame_id = obj.header.frame_id
            box_pose.header.stamp = rospy.Time.now() #- rospy.Duration((12-5)*0.1)
            box_pose.pose = obj.pose
            box_pose_world = self.listener.transformPose(self.world_frame, box_pose)
            converted_obj.header.frame_id = self.world_frame
            converted_obj.pose = box_pose_world.pose
            converted_obj.dimensions = obj.scale
            converted_obj_list.append(converted_obj)
        converted_obj_array.objects = converted_obj_list
        self.objects_list.append(converted_obj_array)
        if(len(self.objects_list) > 10):
            self.objects_list.pop(0)

    def callback_tlr(self, tlr):
        self.tlr = tlr

    def callback_wp(self, waypoints):
        self.waypoint = waypoints

    def update_tf(self):
        for i in range(10):
            #self.listener.waitForTransform("/world", "/base_link", rospy.Time.now(), rospy.Duration(4.0))
            try:
               (trans, rot) = self.listener.lookupTransformFull("/map", rospy.Time.now(), "/base_link", rospy.Time.now() - rospy.Duration((12-i)*0.1),self.world_frame)
            except (tf.LookupException, TypeError, tf.ConnectivityException, tf.ExtrapolationException):
               print "error"
               continue
            e = tf.transformations.euler_from_quaternion(rot)
            q = tf.transformations.quaternion_from_euler(0.0,0.0,e[2])
            self.br.sendTransform(trans, q,rospy.Time.now(),"old_tf_"+str(i),self.world_frame)

    def main(self):
        #rospy.spin()
        
        time.sleep(1.5)
        while not rospy.is_shutdown():
            time.sleep(0.05)
            self.update_tf()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FeatureExtractor')
    parser.add_argument('--devide_flag', default=True, help='Flag of devide feature maps')
    args = parser.parse_args()

    rospy.init_node('FeatureExtractor')
    print rospy.get_rostime()
    FE = FeatureExtractor(args.devide_flag)
    FE.main()
