#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import tf
import csv
import rospy
import rosbag
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Transform
from carla_msgs.msg import CarlaEgoVehicleControl
from extract_feature_image.msg import FeatureImage

# python data_maker.py ../output ../2019-04-27-14-55-08.bag

if len(sys.argv) == 3:
    out_dir = sys.argv[1]
    if not os.path.exists(sys.argv[1]):
        os.makedirs(out_dir)
        out_dir_image = out_dir + '/images'
        os.makedirs(out_dir_image)

    else:
        print sys.argv[1], 'already exists'
        sys.exit()

    bagfile = sys.argv[2]

else:
    print 'illegal arguments exeption.'
    sys.exit()

bridge = CvBridge()


def main():
    image_topic = []#"/occupancy_grid_0",]
    for i in range(10):
       image_topic.append("/occupancy_grid_"+str(i))
       image_topic.append("/obj_bbox_"+str(i))
       image_topic.append("/ego_bbox_"+str(i))
    image_topic.append("/way_image")
    image_topic.append("/map_image")
    image_topic.append("/image_raw")
    tf_topic = '/tf'
    tf_arr = []
    inbag_name = bagfile
    seq = 0
    cmd_topic = '/send_control_command'
    cmd_arr = []
    tl_topic = '/traffic_light'
    tl_arr = []
    image_arr = []

    bag = rosbag.Bag(inbag_name)
    tf_msg = Transform()
    cmd_msg = CarlaEgoVehicleControl()
    for topic, msg, t in bag.read_messages():
        if topic == tf_topic:
            if msg.transforms[0].header.frame_id == "map":
                if msg.transforms[0].child_frame_id == "base_link":
                    tf_msg = msg.transforms[0].transform

        if topic == cmd_topic:
            cmd_msg = msg
        if topic == tl_topic:
            tl_msg = msg
        if topic == "/image_raw":
            image_msg = bridge.imgmsg_to_cv2(msg)

        if topic == "/all":
            for i in range(10):
                if(len(tf_arr)<10):
                   tf_arr.append(tf_msg)
                if(len(image_arr)<10):
                   image_arr.append(image_msg)
            tf_arr.pop(0)
            tf_arr.append(tf_msg)
            image_arr.pop(0)
            image_arr.append(image_msg)

            tf_converted = []

            for i in range(10):
                quaternion = tf_arr[4].rotation
                e = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
                yaw = e[2]
                diff_x = tf_arr[i].translation.x - tf_arr[4].translation.x
                diff_y = tf_arr[i].translation.y - tf_arr[4].translation.y
                x = diff_x*np.cos(-yaw) - diff_y*np.sin(-yaw)
                y = diff_x*np.sin(-yaw) + diff_y*np.cos(-yaw)
                tf_converted.append(x)
                tf_converted.append(y)
                tf_converted.append(yaw)

            with open(out_dir + '/tf.csv', 'ab') as f:
                thewriter = csv.writer(f)
                thewriter.writerow(tf_converted)

            img = image_arr[4]            
            dir_name = "/" + str(seq).zfill(5)
            filename = dir_name + "/image_raw" + '_' +  '.jpg'
            if not os.path.exists(out_dir_image + dir_name):
               os.makedirs(out_dir_image + dir_name)
            cv2.imwrite(out_dir_image + filename, img)

            topic1 = "/occupancy_grid_0"
            img = bridge.imgmsg_to_cv2(msg.occupancy_grid_0)
            dir_name = "/" + str(seq).zfill(5)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            print(out_dir_image + filename)

            topic1 = "/occupancy_grid_1"
            img = bridge.imgmsg_to_cv2(msg.occupancy_grid_1)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)

            topic1 = "/occupancy_grid_2"
            img = bridge.imgmsg_to_cv2(msg.occupancy_grid_2)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)


            topic1 = "/occupancy_grid_3"
            img = bridge.imgmsg_to_cv2(msg.occupancy_grid_3)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)

            topic1 = "/occupancy_grid_4"
            img = bridge.imgmsg_to_cv2(msg.occupancy_grid_4)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)

            topic1 = "/occupancy_grid_5"
            img = bridge.imgmsg_to_cv2(msg.occupancy_grid_5)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/occupancy_grid_6"
            img = bridge.imgmsg_to_cv2(msg.occupancy_grid_6)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/occupancy_grid_7"
            img = bridge.imgmsg_to_cv2(msg.occupancy_grid_7)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/occupancy_grid_8"
            img = bridge.imgmsg_to_cv2(msg.occupancy_grid_8)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/occupancy_grid_9"
            img = bridge.imgmsg_to_cv2(msg.occupancy_grid_9)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)


            topic1 = "/ego_bbox_0"
            img = bridge.imgmsg_to_cv2(msg.ego_bbox_0)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/ego_bbox_1"
            img = bridge.imgmsg_to_cv2(msg.ego_bbox_1)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/ego_bbox_2"
            img = bridge.imgmsg_to_cv2(msg.ego_bbox_2)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/ego_bbox_3"
            img = bridge.imgmsg_to_cv2(msg.ego_bbox_3)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/ego_bbox_4"
            img = bridge.imgmsg_to_cv2(msg.ego_bbox_4)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/ego_bbox_5"
            img = bridge.imgmsg_to_cv2(msg.ego_bbox_5)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/ego_bbox_6"
            img = bridge.imgmsg_to_cv2(msg.ego_bbox_6)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/ego_bbox_7"
            img = bridge.imgmsg_to_cv2(msg.ego_bbox_7)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/ego_bbox_8"
            img = bridge.imgmsg_to_cv2(msg.ego_bbox_8)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/ego_bbox_9"
            img = bridge.imgmsg_to_cv2(msg.ego_bbox_9)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)


            topic1 = "/obj_bbox_0"
            img = bridge.imgmsg_to_cv2(msg.obj_bbox_0)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/obj_bbox_1"
            img = bridge.imgmsg_to_cv2(msg.obj_bbox_1)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/obj_bbox_2"
            img = bridge.imgmsg_to_cv2(msg.obj_bbox_2)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/obj_bbox_3"
            img = bridge.imgmsg_to_cv2(msg.obj_bbox_3)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/obj_bbox_4"
            img = bridge.imgmsg_to_cv2(msg.obj_bbox_4)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/obj_bbox_5"
            img = bridge.imgmsg_to_cv2(msg.obj_bbox_5)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/obj_bbox_6"
            img = bridge.imgmsg_to_cv2(msg.obj_bbox_6)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/obj_bbox_7"
            img = bridge.imgmsg_to_cv2(msg.obj_bbox_7)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/obj_bbox_8"
            img = bridge.imgmsg_to_cv2(msg.obj_bbox_8)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/obj_bbox_9"
            img = bridge.imgmsg_to_cv2(msg.obj_bbox_9)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)



            topic1 = "/ego_points_0"
            img = bridge.imgmsg_to_cv2(msg.ego_points_0)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.flip(img, 1)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/ego_points_1"
            img = bridge.imgmsg_to_cv2(msg.ego_points_1)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.flip(img, 1)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/ego_points_2"
            img = bridge.imgmsg_to_cv2(msg.ego_points_2)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.flip(img, 1)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/ego_points_3"
            img = bridge.imgmsg_to_cv2(msg.ego_points_3)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.flip(img, 1)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/ego_points_4"
            img = bridge.imgmsg_to_cv2(msg.ego_points_4)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.flip(img, 1)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/ego_points_5"
            img = bridge.imgmsg_to_cv2(msg.ego_points_5)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.flip(img, 1)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/ego_points_6"
            img = bridge.imgmsg_to_cv2(msg.ego_points_6)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.flip(img, 1)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/ego_points_7"
            img = bridge.imgmsg_to_cv2(msg.ego_points_7)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.flip(img, 1)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/ego_points_8"
            img = bridge.imgmsg_to_cv2(msg.ego_points_8)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.flip(img, 1)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/ego_points_9"
            img = bridge.imgmsg_to_cv2(msg.ego_points_9)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.flip(img, 1)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)

            topic1 = "/map"
            img = bridge.imgmsg_to_cv2(msg.map)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)
            topic1 = "/way"
            img = bridge.imgmsg_to_cv2(msg.way)
            filename = dir_name + topic1 + '_' +  '.jpg'
            cv2.imwrite(out_dir_image + filename, img)


            seq += 1
            
    bag.close()


if __name__ == "__main__":
    main()
