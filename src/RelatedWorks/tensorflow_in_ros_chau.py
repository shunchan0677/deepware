#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from extract_feature_image.msg import FeatureImage
from sensor_msgs.msg import Joy
from cv_bridge import CvBridge
import tf as tf_ros
import cv2
import numpy as np
import tensorflow as tf
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
from cell2 import ConvLSTMCell


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],
                        padding='SAME')

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1],
                        padding='SAME')

def deconv2d(x,W,size,n2,chan):
    return tf.nn.conv2d_transpose(x, W, tf.stack([chan, size, size, n2]), strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def max_pool_3x3(x):
    return tf.nn.max_pool3d(x, ksize=[1, 1, 2, 2, 1],
                          strides=[1, 1, 2, 2, 1], padding='SAME')


def OGMPred(image):
    with tf.name_scope('enc-conv1') as scope:
        image = tf.reshape(image,[-1]+[400,400]+[1])
        W_conv1 = weight_variable([3, 3, 1, 8])
        b_conv1 = bias_variable([8])
        h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('enc-conv2') as scope:
        W_conv2 = weight_variable([3, 3, 8, 16])
        b_conv2 = bias_variable([16])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('enc-conv3') as scope:
        W_conv3 = weight_variable([3, 3, 16, 32])
        b_conv3 = bias_variable([32])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)


    h_pool3_ = tf.reshape(h_pool3,[-1,10]+[50,50]+[32])

    with tf.name_scope('core-RNN-conv1') as scope:
        cell1 = ConvLSTMCell([50,50], 32, [3, 3])
        outputs1, state1 = tf.nn.dynamic_rnn(cell1, h_pool3_, dtype=h_pool3_.dtype)


    outputs_ = tf.reshape(outputs1,[-1]+[50,50]+[32])

    with tf.name_scope('dec-conv1') as scope:
        W_conv4 = weight_variable([3, 3, 16, 32])
        b_conv4 = bias_variable([16])
        h_conv4 = tf.nn.relu(deconv2d(outputs_, W_conv4,100,16,10) + b_conv4)

    with tf.name_scope('dec-conv2') as scope:
        W_conv5 = weight_variable([3, 3, 8, 16]) 
        b_conv5 = bias_variable([8])
        h_conv5 = tf.nn.relu(deconv2d(h_conv4, W_conv5, 200,8,10) + b_conv5)

    with tf.name_scope('dec-conv3') as scope:
        W_conv6 = weight_variable([3, 3, 1, 8])
        b_conv6 = bias_variable([1])
        h_conv6 = tf.sigmoid(deconv2d(h_conv5, W_conv6, 400,1,10) + b_conv6)

    return h_conv6


def AgentNet(ogm, ego_bbox, ego_p, maps, keep_prob):
    ogm = tf.reshape(ogm,[10,400,400,1])
    ego_bbox = tf.reshape(ego_bbox,[10,400,400,1])
    ego_p = tf.reshape(ego_p,[10,400,400,1])
    maps = tf.reshape(maps,[2,400,400,1])

    for i in range(10):
		inputs = tf.concat([[ogm[i,:,:,:]],[ego_bbox[i,:,:,:]]],0)
		inputs = tf.concat([inputs,[ego_p[i,:,:,:]]],0)
		inputs = tf.concat([inputs,maps],0)
		if(i==0):
			inputs_0 = inputs
		else:
			inputs_0 = tf.concat([inputs_0,inputs],3)

    with tf.name_scope('age-enc-conv1') as scope:
        image = tf.reshape(inputs_0,[-1]+[400,400]+[5])
        W_conv1 = weight_variable([3, 3, 5, 8])
        b_conv1 = bias_variable([8])
        h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('age-enc-conv2') as scope:
        W_conv2 = weight_variable([3, 3, 8, 16])
        b_conv2 = bias_variable([16])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('age-enc-conv3') as scope:
        W_conv3 = weight_variable([3, 3, 16, 32])
        b_conv3 = bias_variable([32])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)


    h_pool3_ = tf.reshape(h_pool3,[-1,10]+[50,50]+[32])

    with tf.variable_scope('age-core-RNN-conv1') as scope:
        cell1 = ConvLSTMCell([50,50], 32, [3, 3],"age")
        outputs1, state1 = tf.nn.dynamic_rnn(cell1, h_pool3_, dtype=h_pool3_.dtype)


    outputs_ = tf.reshape(outputs1,[-1]+[50,50]+[32])

    with tf.name_scope('age-dec-conv1') as scope:
        W_conv4 = weight_variable([3, 3, 16, 32])
        b_conv4 = bias_variable([16])
        h_conv4 = tf.nn.relu(deconv2d(outputs_, W_conv4,100,16,10) + b_conv4)

    with tf.name_scope('age-dec-conv2') as scope:
        W_conv5 = weight_variable([3, 3, 8, 16]) 
        b_conv5 = bias_variable([8])
        h_conv5 = tf.nn.relu(deconv2d(h_conv4, W_conv5, 200,8,10) + b_conv5)

    with tf.name_scope('age-dec-conv3') as scope:
        W_conv6 = weight_variable([3, 3, 2, 8])
        b_conv6 = bias_variable([2])
        h_conv6 = deconv2d(h_conv5, W_conv6, 400,2,10) + b_conv6



    with tf.name_scope('age-enc-conv7') as scope:
        W_conv7 = weight_variable([3, 3, 32, 48])
        b_conv7 = bias_variable([48])
        h_conv7 = tf.nn.relu(conv2d(outputs_, W_conv7) + b_conv7)
        h_pool7 = max_pool_2x2(h_conv7)

    with tf.name_scope('age-enc-conv8') as scope:
        W_conv8 = weight_variable([3, 3, 48, 64])
        b_conv8 = bias_variable([64])
        h_conv8 = tf.nn.relu(conv2d(h_pool7, W_conv8) + b_conv8)
        h_pool8 = max_pool_2x2(h_conv8)


    with tf.name_scope('age-enc-conv9') as scope:
        W_conv9 = weight_variable([3, 3, 64, 64])
        b_conv9 = bias_variable([64])
        h_conv9 = tf.nn.relu(conv2d(h_pool8, W_conv9) + b_conv9)
        h_pool9 = max_pool_2x2(h_conv9)


    h_conv6_flatten = tf.reshape(h_pool9, [10, 7 * 7 * 64])

    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([ 7 * 7 * 64, 1164])
        b_fc1 = bias_variable([1164])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv6_flatten, W_fc1) + b_fc1)

    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1164, 100])
        b_fc2 = bias_variable([100])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    with tf.name_scope('fc3') as scope:
        W_fc3 = weight_variable([100, 3])
        b_fc3 = bias_variable([3])
        y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3


    h_conv5_d1 = tf.reshape(h_conv6, [ 2, 400, 400 ,10])

    p_map = h_conv5_d1[0]

    #p_map_soft = tf.nn.softmax(tf.reshape(p_map,[1,400*400,10]))
    p_map_soft = tf.exp(p_map) / tf.reduce_sum(tf.reduce_sum(tf.exp(p_map), 0),0)
    p_out = tf.reshape(p_map_soft,[1,10,400,400,1])

    return tf.reshape(y_conv,[10,3]), tf.sigmoid(h_conv5_d1[1]) ,p_out

class RosTensorFlow():
    def __init__(self):
	self.sub_image0_flag=0
	self.sub_twist_flag=0
        self._cv_bridge = CvBridge()
        self._sub1 = rospy.Subscriber('/all', FeatureImage, self.callback, queue_size=1)
        self._pub = rospy.Publisher('/twist_raw', TwistStamped, queue_size=1)


        self.OGM_x = tf.placeholder(tf.float32, [1,10] + [400, 400]+ [1])
        self.ego_bbox_x = tf.placeholder(tf.float32, [1,10] + [400, 400]+ [1])
        self.ego_p_x = tf.placeholder(tf.float32, [1,10] + [400, 400]+ [1])
        self.maps_x = tf.placeholder(tf.float32, [1,2] + [400, 400]+ [1])

        self.OGM_y1 = tf.placeholder(tf.float32, [1,10] + [400, 400]+ [1])
        self.ego_bbox_y = tf.placeholder(tf.float32, [1,10] + [400, 400]+ [1])
        self.ego_p_y = tf.placeholder(tf.float32, [1,10] + [400, 400]+ [1])
        self.y_r = tf.placeholder(tf.float32, [10,3])
        self.keep_prob = tf.placeholder(tf.float32)

        self.OGM_pred = OGMPred(self.OGM_x)
        self.OGM_pred1 = tf.where(self.OGM_pred > 0.1,tf.ones([10,400,400,1]),tf.zeros([10,400,400,1]))
        self.OGM_y = tf.where(self.OGM_y1 > 0.1,tf.ones([1,10,400,400,1]),tf.zeros([1,10,400,400,1]))
        self.y_conv,self.output_bbox_map,self.output_pose_map = AgentNet(self.OGM_pred1, self.ego_bbox_x, self.ego_p_x, self.maps_x, self.keep_prob)

        self._session = tf.InteractiveSession()
        self._saver = tf.train.Saver()
        self._session.run(tf.global_variables_initializer())
        self._saver.restore(self._session, "./zoo/20200103ae_sim_ours_carla_true_y_10xmap_wp_copy_obj2.ckpt")

        self.listener = tf_ros.TransformListener()

    def callback(self,msg):
       flag = True #(self.sub_way_flag==1 and self.sub_map_flag==1 and self.sub_grid9_flag==1 and self.sub_grid8_flag==1 and self.sub_grid7_flag== 1 and self.sub_grid6_flag==1 and self.sub_grid5_flag==1 and self.sub_bbox5_flag==1 and self.sub_bbox6_flag==1 and self.sub_bbox7_flag==1 and self.sub_bbox8_flag==1 and self.sub_bbox9_flag==1)
       if(flag):
            """
            try:
                time = rospy.Time.now()
                self.listener.waitForTransform("/base_link", "/map", time, rospy.Duration(1.0))
                (trans,rot) = self.listener.lookupTransform('/base_link', '/map', time)
                (trans1,rot1) = self.listener.lookupTransform('/base_link', '/map', time-rospy.Duration(0.1))
                (trans2,rot2) = self.listener.lookupTransform('/base_link', '/map', time-rospy.Duration(0.2))
                (trans3,rot3) = self.listener.lookupTransform('/base_link', '/map', time-rospy.Duration(0.3))
                (trans4,rot4) = self.listener.lookupTransform('/base_link', '/map', time-rospy.Duration(0.4))
            except (tf_ros.LookupException, tf_ros.ConnectivityException, tf_ros.ExtrapolationException):
                return 0
                print("Fail tf")


            list_d = []
            list_d.append(trans1)
            list_d.append(trans2)
            list_d.append(trans3)
            list_d.append(trans4)
            res_d = []
            e = tf_ros.transformations.euler_from_quaternion(rot)
            yaw = e[2]
            for tra in list_d:
               diff_x = tra[0] - trans[0]
               diff_y = tra[1] - trans[1]
               x = diff_x*np.cos(-yaw) - diff_y*np.sin(-yaw)
               y = diff_x*np.sin(-yaw) + diff_y*np.cos(-yaw)
               res_d.append([x,y])
            """          
            
            #print trans
            self.sub_way_flag=0
            self.sub_map_flag=0
            self.sub_grid9_flag=0
            self.sub_grid8_flag=0
            self.sub_grid7_flag=0
            self.sub_grid6_flag=0
            self.sub_grid5_flag=0
            self.sub_bbox5_flag=0
            self.sub_bbox6_flag=0
            self.sub_bbox7_flag=0
            self.sub_bbox8_flag=0
            self.sub_bbox9_flag=0

            #self.sub_image0_flag=0
            #self.sub_twist_flag=0
            list_image = []
            map_image = self._cv_bridge.imgmsg_to_cv2(msg.map, "mono8")
            way_image = self._cv_bridge.imgmsg_to_cv2(msg.way, "mono8")
            grid9_image = self._cv_bridge.imgmsg_to_cv2(msg.occupancy_grid_9, "mono8")
            grid8_image = self._cv_bridge.imgmsg_to_cv2(msg.occupancy_grid_8, "mono8")
            grid7_image = self._cv_bridge.imgmsg_to_cv2(msg.occupancy_grid_7, "mono8")
            grid6_image = self._cv_bridge.imgmsg_to_cv2(msg.occupancy_grid_6, "mono8")
            grid5_image = self._cv_bridge.imgmsg_to_cv2(msg.occupancy_grid_5, "mono8")


            ego9_image = self._cv_bridge.imgmsg_to_cv2(msg.ego_bbox_9, "mono8")
            ego8_image = self._cv_bridge.imgmsg_to_cv2(msg.ego_bbox_8, "mono8")
            ego7_image = self._cv_bridge.imgmsg_to_cv2(msg.ego_bbox_7, "mono8")
            ego6_image = self._cv_bridge.imgmsg_to_cv2(msg.ego_bbox_6, "mono8")
            ego5_image = self._cv_bridge.imgmsg_to_cv2(msg.ego_bbox_5, "mono8")


            egop9_image = self._cv_bridge.imgmsg_to_cv2(msg.ego_points_9, "mono8")
            egop8_image = self._cv_bridge.imgmsg_to_cv2(msg.ego_points_8, "mono8")
            egop7_image = self._cv_bridge.imgmsg_to_cv2(msg.ego_points_7, "mono8")
            egop6_image = self._cv_bridge.imgmsg_to_cv2(msg.ego_points_6, "mono8")
            egop5_image = self._cv_bridge.imgmsg_to_cv2(msg.ego_points_5, "mono8")

            test_image_result = np.zeros((1,32,400,400,1))
            

            test_image_result[0,0,:,:,0] = grid5_image
            test_image_result[0,1,:,:,0] = grid6_image
            test_image_result[0,2,:,:,0] = grid7_image
            test_image_result[0,3,:,:,0] = grid8_image
            test_image_result[0,4,:,:,0] = grid9_image

            test_image_result[:,5,:,:,:] = np.zeros((1,400,400,1))
            test_image_result[:,6,:,:,:] = np.zeros((1,400,400,1))
            test_image_result[:,7,:,:,:] = np.zeros((1,400,400,1))
            test_image_result[:,8,:,:,:] = np.zeros((1,400,400,1))
            test_image_result[:,9,:,:,:] = np.zeros((1,400,400,1))

            test_image_result[0,10,:,:,0] = ego5_image
            test_image_result[0,11,:,:,0] = ego6_image
            test_image_result[0,12,:,:,0] = ego7_image
            test_image_result[0,13,:,:,0] = ego8_image
            test_image_result[0,14,:,:,0] = ego9_image



            test_image_result[:,15,:,:,:] = np.zeros((1,400,400,1))
            test_image_result[:,16,:,:,:] = np.zeros((1,400,400,1))
            test_image_result[:,17,:,:,:] = np.zeros((1,400,400,1))
            test_image_result[:,18,:,:,:] = np.zeros((1,400,400,1))
            test_image_result[:,19,:,:,:] = np.zeros((1,400,400,1))


            test_image_result[0,20,:,:,0] = egop5_image
            test_image_result[0,21,:,:,0] = egop6_image
            test_image_result[0,22,:,:,0] = egop7_image
            test_image_result[0,23,:,:,0] = egop8_image
            test_image_result[0,24,:,:,0] = egop9_image

            #test_image_result[0,21,np.max(int(-res_d[0][0]* 400.0 / 80.0 + 4 * 400 / 5),0),np.max(int(-res_d[0][1]*  400 / 80.0 +200),0),0] = 255

            #test_image_result[0,22,np.max(int(-res_d[1][0]* 400.0 / 80.0 + 4 * 400 / 5),0),np.max(int(-res_d[1][1]*  400 / 80.0 +200),0),0] = 255
            #test_image_result[0,23,np.max(int(-res_d[2][0]* 400.0 / 80.0 + 4 * 400 / 5),0),np.max(int(-res_d[2][1]*  400 / 80.0 +200),0),0] = 255
            #test_image_result[0,24,np.max(int(-res_d[3][0]* 400.0 / 80.0 + 4 * 400 / 5),0),np.max(int(-res_d[3][1]*  400 / 80.0 +200),0),0] = 255

            test_image_result[:,25,:,:,:] = np.zeros((1,400,400,1))
            test_image_result[:,26,:,:,:] = np.zeros((1,400,400,1))
            test_image_result[:,27,:,:,:] = np.zeros((1,400,400,1))
            test_image_result[:,28,:,:,:] = np.zeros((1,400,400,1))
            test_image_result[:,29,:,:,:] = np.zeros((1,400,400,1))
            test_image_result[0,30,:,:,0] = map_image
            test_image_result[0,31,:,:,0] = way_image
            test_image_result = test_image_result/255.0

            result = self._session.run(self.y_conv, feed_dict={self.OGM_x:test_image_result[:,0:10,:,:,:],self.ego_bbox_x:test_image_result[:,10:20,:,:,:],self.ego_p_x:test_image_result[:,20:30,:,:,:],self.maps_x:test_image_result[:,30:,:,:,:]*10.0, self.keep_prob: 1.0})
            print result
            x = result[9][0]
            y = result[9][1]
            l = x*x + y*y
            if(abs(x) < 0.1):
                self._stre = 0.0
            else:
                self._stre = -y/l
            self._ac = x
            twist = TwistStamped()
            stre = self._stre
            ac = self._ac * 2.0
            twist.twist.linear.x = ac 
            twist.twist.angular.z = stre *ac
            self._pub.publish(twist)

	    print stre



    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('rostensorflow')
    tensor = RosTensorFlow()
    tensor.main()
