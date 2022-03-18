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

        if topic in image_topic:
            img = bridge.imgmsg_to_cv2(msg)
            dir_name = "/" + str(msg.header.seq).zfill(5)
            if topic == "/image_raw":
                dir_name = "/" + str(seq).zfill(5)
                seq += 1
            filename = dir_name + topic + '_' +  '.jpg'
            if not os.path.exists(out_dir_image + dir_name):
               os.makedirs(out_dir_image + dir_name)
            
            cv2.imwrite(out_dir_image + filename, img)
            print(out_dir_image + filename)

        if topic == "/occupancy_grid_5":
            for i in range(10):
                if(len(tf_arr)<10):
                   tf_arr.append(tf_msg)
                if(len(cmd_arr)<10):
                   cmd_arr.append(cmd_msg)
            tf_arr.pop(0)
            tf_arr.append(tf_msg)
            cmd_arr.pop(0)
            cmd_arr.append(cmd_msg)

            tf_converted = []

            for i in range(10):
                quaternion = tf_arr[i].rotation
                e = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
                yaw = e[2]
                diff_x = tf_arr[i].translation.x - tf_arr[4].translation.x
                diff_y = tf_arr[i].translation.y - tf_arr[4].translation.y
                x = diff_x*np.cos(yaw) - diff_y*np.sin(yaw)
                y = diff_x*np.sin(yaw) + diff_y*np.cos(yaw)
                tf_converted.append(x)
                tf_converted.append(y)
                tf_converted.append(yaw)

            with open(out_dir + '/tf.csv', 'ab') as f:
                thewriter = csv.writer(f)
                thewriter.writerow(tf_converted)

            with open(out_dir + '/cmd.csv', 'ab') as f_a:
                thewriter = csv.writer(f_a)
                cmd_list = []
                for i in range(10):
                    cmd_list.append(cmd_arr[i].throttle)
                    cmd_list.append(cmd_arr[i].brake)
                    cmd_list.append(cmd_arr[i].steer)
                thewriter.writerow(cmd_list)
    bag.close()


if __name__ == "__main__":
    main()
