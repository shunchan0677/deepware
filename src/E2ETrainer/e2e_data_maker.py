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

def rotation_(target_tf_msg,tf_msg5):
     e = tf.transformations.euler_from_quaternion((tf_msg5.rotation.x,tf_msg5.rotation.y,tf_msg5.rotation.z,tf_msg5.rotation.w))
     yaw5 = e[2]

     x_d = target_tf_msg.translation.x - tf_msg5.translation.x
     y_d = target_tf_msg.translation.y - tf_msg5.translation.y
     x_r = x_d*np.cos(yaw5) + y_d*np.sin(yaw5)
     y_r = -x_d*np.sin(yaw5) + y_d*np.cos(yaw5)
     e = tf.transformations.euler_from_quaternion((target_tf_msg.rotation.x,target_tf_msg.rotation.y,target_tf_msg.rotation.z,target_tf_msg.rotation.w))
     yaw = e[2] - yaw5
     return x_r,y_r,yaw


def main():
    image_topic = []
    for i in range(10):
       image_topic.append("/occupancy_grid_"+str(i))
    tf_topic = '/tf'
    feature_map_topics = []
    feature_map_topics.append("/vector_image_raw/ego_vehicle")
    feature_map_topics.append("/vector_image_raw/hd_map")
    feature_map_topics.append("/vector_image_raw/objects")
    feature_map_topics.append("/vector_image_raw/waypoint")
    feature_map_topics.append("/vector_image_raw")
    tf_arr = []
    inbag_name = bagfile
    save_flag = False
    seq = 0

    bag = rosbag.Bag(inbag_name)
    tf_msg_base_link = Transform()
    for topic, msg, t in bag.read_messages():
        if topic == tf_topic:
            for data in msg.transforms:
                if data.header.frame_id == "/map":
                    if data.child_frame_id == "/base_link":
                        tf_msg_base_link = data.transform
                        print data.child_frame_id

        if topic in image_topic:
            img = bridge.imgmsg_to_cv2(msg)
            dir_name = "/" + str(msg.header.seq).zfill(5)
            if topic == "/occupancy_grid_5":
                dir_name = "/" + str(seq).zfill(5)
                seq += 1
            filename = dir_name + topic + '_' +  '.jpg'
            if not os.path.exists(out_dir_image + dir_name):
               os.makedirs(out_dir_image + dir_name)
            
            if(save_flag):
                cv2.imwrite(out_dir_image + filename, img)
                print(out_dir_image + filename)

        if topic in feature_map_topics:
            img = bridge.imgmsg_to_cv2(msg)
            dir_name = "/" + str(msg.header.seq).zfill(5)
            if topic == "/vector_image_raw":
                dir_name = "/" + str(seq).zfill(5)
            filename = dir_name + "/" + topic.replace("/","_") + '_' +  '.jpg'
            if not os.path.exists(out_dir_image + dir_name):
               os.makedirs(out_dir_image + dir_name)
            
            img = cv2.resize(img , (256, 256))
            if(save_flag):
                cv2.imwrite(out_dir_image + filename, img)
                print(out_dir_image + filename)

        if topic == "/occupancy_grid_5":

            if(save_flag):
                with open(out_dir + '/tf.csv', 'ab') as f:
                    thewriter = csv.writer(f)
                    thewriter.writerow(tf_result)

            tf_arr.append(tf_msg_base_link)
            if(len(tf_arr)>5):
               save_flag = True
               tf_arr.pop(0)
               tf_result = []
               for tfs in tf_arr:
                   x,y,yaw = rotation_(tfs,tf_arr[0])
                   tf_result += [x,y,yaw]
            
    bag.close()


if __name__ == "__main__":
    main()
