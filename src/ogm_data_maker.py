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


def main():
    image_topic = []#"/occupancy_grid_0",]
    for i in range(10):
       image_topic.append("/occupancy_grid_"+str(i))
    tf_topic = '/tf'
    tf_arr = []
    inbag_name = bagfile
    seq = 0

    bag = rosbag.Bag(inbag_name)
    tf_msg = Transform()
    for topic, msg, t in bag.read_messages():
        if topic == tf_topic:
            if msg.transforms[0].header.frame_id == "/map":
                if msg.transforms[0].child_frame_id == "/old_tf_5":
                    tf_msg = msg.transforms[0].transform

        if topic in image_topic:
            img = bridge.imgmsg_to_cv2(msg)
            dir_name = "/" + str(msg.header.seq).zfill(5)
            if topic == "/occupancy_grid_1":
                dir_name = "/" + str(seq).zfill(5)
                seq += 1
            filename = dir_name + topic + '_' +  '.jpg'
            if not os.path.exists(out_dir_image + dir_name):
               os.makedirs(out_dir_image + dir_name)
            
            cv2.imwrite(out_dir_image + filename, img)
            print(out_dir_image + filename)

        if topic == "/occupancy_grid_5":
            with open(out_dir + '/tf.csv', 'ab') as f:
                thewriter = csv.writer(f)
                thewriter.writerow([tf_msg.translation.x, tf_msg.translation.y, tf_msg.translation.z,
                                    tf_msg.rotation.x, tf_msg.rotation.y, tf_msg.rotation.z, tf_msg.rotation.w])
    bag.close()


if __name__ == "__main__":
    main()
