#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np
import random
import glob
import threading
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from cell2 import ConvLSTMCell
import copy
import os
import cv2

class BackgroundGenerator(threading.Thread):
    """Background generator
    Reference:
        https://stackoverflow.com/questions/7323664/python-generator-pre-fetch
    Args:
        generator (object): generator instance
        max_prefetch (int): max number of prefetch
    """

    def __init__(self, generator, max_prefetch=1):
        threading.Thread.__init__(self)
        if sys.version_info.major == 2:
            from Queue import Queue
        else:
            from queue import Queue
        self.queue = Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()
        print("backgrounder successfully started")

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class GeneratorBuilder:
    def __init__(self, image_label, r_label, batch_size):
        self.train_image = []
        self.train_r = []
        self.image_label = image_label
        self.r_label = r_label
        self.batch_size = batch_size
        self.num_flame = 10

    def build(self):
        while True:
            self.random_seq = list(range(len(self.image_label)-1))
            random.shuffle(self.random_seq)
            for l in range(len(self.image_label)-1):
                flag = True
                image = np.zeros((13,256,256,1))
                for flame in range(self.num_flame):
                    path = self.image_label[self.random_seq[l]]+"/bbox_"+str(flame)+".png"
                    if(os.path.exists(path) == False):
                        flag = False
                path = self.image_label[self.random_seq[l]]+"/ego_pose.png"
                if(os.path.exists(path) == False):
                    flag = False
                path = self.image_label[self.random_seq[l]]+"/map.png"
                if(os.path.exists(path) == False):
                    flag = False
                path = self.image_label[self.random_seq[l]]+"/waypoint.png"
                if(os.path.exists(path) == False):
                    flag = False

                if(flag):
                    for flame in range(self.num_flame):
                        image[flame,:,:,:] = np.asarray(cv2.imread(self.image_label[self.random_seq[l]]+"/lidar_"+str(flame)+".png",cv2.IMREAD_GRAYSCALE)).reshape((1,256,256,1))/ 255.0
                    image[10,:,:,:] = np.asarray(cv2.imread(self.image_label[self.random_seq[l]]+"/ego_pose.png",cv2.IMREAD_GRAYSCALE)).reshape((1,256,256,1))/ 255.0
                    image[11,:,:,:] = np.asarray(cv2.imread(self.image_label[self.random_seq[l]]+"/map.png",cv2.IMREAD_GRAYSCALE)).reshape((1,256,256,1))/ 255.0
                    image[12,:,:,:] = np.asarray(cv2.imread(self.image_label[self.random_seq[l]]+"/waypoint.png",cv2.IMREAD_GRAYSCALE)).reshape((1,256,256,1))/ 255.0
                    self.train_image.append(image)
                    self.train_r.append(self.r_label[self.random_seq[l]][12:15])
                    if((l + 1) % self.batch_size == 0):
                        self.train_image = np.asarray(self.train_image).reshape((-1,self.num_flame+3,256,256,1))
                        #self.train_r = copy.copy(self.train_image)
                        #self.train_image[:,5,:,:,:] = np.zeros((1,256,256,1))
                        #self.train_image[:,6,:,:,:] = np.zeros((1,256,256,1))
                        #self.train_image[:,7,:,:,:] = np.zeros((1,256,256,1))
                        #self.train_image[:,8,:,:,:] = np.zeros((1,256,256,1))
                        #self.train_image[:,9,:,:,:] = np.zeros((1,256,256,1))
                        yield [self.train_image, self.train_r]
                        self.train_image = []
                        self.train_r = []
            yield None


def make_datalist(filenames):
    train_image_label = []
    train_r_label = []

    for filename in filenames:
        image_file = glob.glob("/home/s_seiya/tmp/lyft_data/" +
                               filename + "/*")
        image_file.sort()
        train_image_label+=image_file

        r = np.loadtxt("/home/s_seiya/tmp/lyft_data/" + filename +
                       ".csv", delimiter=",").tolist()
        train_r_label += r[:len(image_file)]

    print(len(train_image_label),len(train_r_label))
    return train_image_label, train_r_label


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
        image = tf.reshape(image,[-1]+[256,256]+[1])
        W_conv1 = weight_variable([3, 3, 1, 16])
        b_conv1 = bias_variable([16])
        h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('enc-conv2') as scope:
        W_conv2 = weight_variable([3, 3, 16, 32])
        b_conv2 = bias_variable([32])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_ = tf.reshape(h_pool2,[-1,10]+[64,64]+[32])

    with tf.name_scope('core-RNN-conv1') as scope:
        cell1 = ConvLSTMCell([64,64], 32, [3, 3])
        outputs1, state1 = tf.nn.dynamic_rnn(cell1, h_pool2_, dtype=h_pool2_.dtype)


    outputs_ = tf.reshape(outputs1,[-1]+[64,64]+[32])

    with tf.name_scope('dec-conv1') as scope:
        W_conv3 = weight_variable([3, 3, 16, 32])
        b_conv3 = bias_variable([16])
        h_conv3 = tf.nn.relu(deconv2d(outputs_, W_conv3,128,16,10) + b_conv3)

    with tf.name_scope('dec-conv2') as scope:
        W_conv4 = weight_variable([3, 3, 1, 16]) 
        b_conv4 = bias_variable([1])
        h_conv4 = tf.sigmoid(deconv2d(h_conv3, W_conv4,256,1,10) + b_conv4) 

    return h_conv4


def AgentNet(ogm, image, keep_prob):
    ogm1 = tf.reshape(ogm,[-1,10,256,256,1])
    inputs = tf.concat([ogm1,image],1)
    inputs = tf.reshape(inputs,[-1]+[256,256]+[1])
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 1, 24])
        b_conv1 = bias_variable([24])
        h_conv1 = tf.nn.relu(conv2d(inputs, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 24, 36])
        b_conv2 = bias_variable([36])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('conv3') as scope:
        W_conv3 = weight_variable([5, 5, 36, 48])
        b_conv3 = bias_variable([48])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)

    with tf.name_scope('conv4') as scope:
        W_conv4 = weight_variable([3, 3, 48, 64])
        b_conv4 = bias_variable([64])
        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
        h_pool4 = max_pool_2x2(h_conv4)

    with tf.name_scope('conv5') as scope:
        W_conv5 = weight_variable([3, 3, 64, 64])
        b_conv5 = bias_variable([64])
        h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
        h_pool5 = max_pool_2x2(h_conv5)

    h_conv5_flatten = tf.reshape(h_pool5, [1, 13 * 8 * 8 * 64])

    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([13 * 8 * 8 * 64, 1164])
        b_fc1 = bias_variable([1164])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flatten, W_fc1) + b_fc1)

    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1164, 100])
        b_fc2 = bias_variable([100])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    with tf.name_scope('fc3') as scope:
        W_fc3 = weight_variable([100, 3])
        b_fc3 = bias_variable([3])
        y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3

    h_fc1_d_flatten = tf.reshape(h_pool3, [ 1, 32, 32 ,48*13])


    with tf.name_scope('de-conv4') as scope:
        W_conv4_d = weight_variable([5, 5, 2, 48*13])
        b_conv4_d = bias_variable([2])
        h_conv4_d = deconv2d(h_fc1_d_flatten, W_conv4_d,64,2,1) + b_conv4_d


    h_conv5_d1 = tf.reshape(h_conv4_d, [ 2, 64, 64 ,1])

    p_map = h_conv5_d1[0]

    p_map_soft = tf.nn.softmax(tf.reshape(p_map,[-1,64*64]))
    p_out = tf.reshape(p_map_soft,[64,64,1])

    return y_conv, tf.sigmoid(h_conv5_d1[1]) ,p_out


if __name__ == '__main__':
    print("ChauffeurNet anno training result log")
    tf.reset_default_graph()
    l = 0
    basermse = 1000
    batchSize = 1

    OGM_x = tf.placeholder(tf.float32, [None,10] + [256, 256]+ [1])
    image1 = tf.placeholder(tf.float32, [None,3] + [256, 256]+ [1])
    OGM_y1 = tf.placeholder(tf.float32, [None,10] + [256, 256]+ [1])
    y_r = tf.placeholder(tf.float32, [None,3])
    keep_prob = tf.placeholder(tf.float32)

    OGM_pred = OGMPred(OGM_x)
    OGM_pred1 = tf.where(OGM_pred > 0.1,tf.ones([10,256,256,1]),tf.zeros([10,256,256,1]))
    OGM_y = tf.where(OGM_y1 > 0.1,tf.ones([1,10,256,256,1]),tf.zeros([1,10,256,256,1]))
    y_conv,output_bbox_map,output_pose_map = AgentNet(OGM_y, image1, keep_prob)
    print (output_pose_map)

    result_x = y_r[0][1]#result[0]
    result_y = y_r[0][0]#result[1]


    l2 = result_x**2 + result_y**2
    r = l2 / (2*result_y)
    yaw = y_r[0][2]#tf.math.atan(result_x/(r-result_y))#y_r[0][2] #

    image_center_point = tf.constant([[400 / 2, int(4 * 400 / 5.0)],[400 / 2, int(4 * 400 / 5.0)],[400 / 2, int(4 * 400 / 5.0)],[400 / 2, int(4 * 400 / 5.0)]])

    box_position_velo = [-result_x, result_y]

    rot_mat = [[tf.math.cos(yaw), -tf.math.sin(yaw)], [tf.math.sin(yaw), tf.math.cos(yaw)]]
    box_size = tf.constant([1.695 / 2.0, 4.36 / 2.0])
    box_mat = tf.constant([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])
    box_4 = box_size * box_mat
    rot_res = tf.matmul(box_4, rot_mat)

    rot_res_rated = (box_position_velo + rot_res) * 400 / 80.0
    image_box = tf.subtract(tf.cast(image_center_point,tf.int32), tf.cast(rot_res_rated,tf.int32))

    max_x = tf.math.reduce_max(image_box[:, 0])
    max_y = tf.math.reduce_max(image_box[:, 1])
    min_x = tf.math.reduce_min(image_box[:, 0])
    min_y = tf.math.reduce_min(image_box[:, 1])

    clip_tf = tf.math.logical_and(tf.math.logical_and((min_x > 0),(max_x <  400)),tf.math.logical_and((min_y > 0),(max_y <  400))) 

    row_idx,col_idx = tf.meshgrid(tf.range(400),tf.range(400))

    image_box_c = tf.clip_by_value(image_box,0,400)

    d_x = image_box_c[1,0] - image_box_c[0,0]
    d_y = image_box_c[1,1] - image_box_c[0,1]
    condition = tf.cast(tf.math.abs(d_x) > tf.constant(0),tf.bool)
    d_x1 = tf.where(condition, tf.cast(d_x,tf.float32), 1.0)
    d_y1 = tf.where(condition, tf.cast(d_y,tf.float32), 1.0)
    alpha = tf.cond(condition, lambda:d_y1/ d_x1, lambda: tf.constant(-1.0, dtype=tf.float32))
    beta = tf.where(condition, tf.cast(image_box_c[0,1],tf.float32) - alpha*tf.cast(image_box_c[0,0],tf.float32), tf.cast(image_box_c[0,0],tf.float32))
    beta2 = tf.where(condition, tf.cast(image_box_c[2,1],tf.float32) - alpha*tf.cast(image_box_c[2,0],tf.float32), tf.cast(image_box_c[2,0],tf.float32))
    bool_mat_0_1 = tf.where(condition, tf.cast(col_idx,tf.float32)-tf.cast(row_idx,tf.float32)*alpha, tf.cast(row_idx,tf.float32))
    map_1 = tf.where(bool_mat_0_1 > beta,tf.ones([400,400]), tf.zeros([400,400]))
    map_2 = tf.where(bool_mat_0_1 < beta2,tf.ones([400,400]), tf.zeros([400,400]))
    bool_tf_0_1_t = (map_1*map_2)
    map_1 = tf.where(bool_mat_0_1 < beta,tf.ones([400,400]), tf.zeros([400,400]))
    map_2 = tf.where(bool_mat_0_1 > beta2,tf.ones([400,400]), tf.zeros([400,400]))
    bool_tf_0_1_f = (map_1*map_2)

    condition2 = tf.cast(alpha < tf.constant(0.0),tf.bool)
    bool_tf_0_1 = tf.where(condition2, bool_tf_0_1_t, bool_tf_0_1_f)
    condition3_0 = tf.cast(tf.equal(alpha,tf.constant(0.0)),tf.bool)
    condition3_1 = tf.cast(d_x < tf.constant(0),tf.bool)
    condition3 = tf.math.logical_and(condition3_0,condition3_1)
    bool_tf_0_1 = tf.where(condition3, bool_tf_0_1_t, bool_tf_0_1)

    d_x_l = image_box_c[2,0] - image_box_c[1,0]
    d_y_l = image_box_c[2,1] - image_box_c[1,1]
    condition_l = tf.cast(tf.math.abs(d_x_l) > tf.constant(0),tf.bool)
    d_x1_l = tf.where(condition_l, tf.cast(d_x_l,tf.float32), 1.0)
    d_y1_l = tf.where(condition_l, tf.cast(d_y_l,tf.float32), 1.0)
    alpha_l = tf.cond(condition_l, lambda:d_y1_l/ d_x1_l, lambda: tf.constant(1.0, dtype=tf.float32))
    beta_l = tf.where(condition_l, tf.cast(image_box_c[1,1],tf.float32) - alpha_l*tf.cast(image_box_c[1,0],tf.float32), tf.cast(image_box_c[1,0],tf.float32))
    beta2_l = tf.where(condition_l, tf.cast(image_box_c[0,1],tf.float32) - alpha_l*tf.cast(image_box_c[0,0],tf.float32), tf.cast(image_box_c[0,0],tf.float32))
    bool_mat_0_1_l = tf.where(condition_l, tf.cast(col_idx,tf.float32)-tf.cast(row_idx,tf.float32)*alpha_l, tf.cast(row_idx,tf.float32))
    map_1_l = tf.where(bool_mat_0_1_l > beta_l,tf.ones([400,400]), tf.zeros([400,400]))
    map_2_l = tf.where(bool_mat_0_1_l < beta2_l,tf.ones([400,400]), tf.zeros([400,400]))
    bool_tf_0_1_t_l = (map_1_l*map_2_l)
    map_1_l = tf.where(bool_mat_0_1_l < beta_l,tf.ones([400,400]), tf.zeros([400,400]))
    map_2_l = tf.where(bool_mat_0_1_l > beta2_l,tf.ones([400,400]), tf.zeros([400,400]))
    bool_tf_0_1_f_l = (map_1_l*map_2_l)

    condition2_0_l = tf.cast(tf.equal(condition_l,tf.cast(False,tf.bool)),tf.bool)
    condition2_1_l = tf.cast(d_y_l >= tf.constant(0),tf.bool)
    condition2_l = tf.math.logical_and(condition2_0_l,condition2_1_l)
    bool_tf_0_1_l = tf.where(condition2_l, bool_tf_0_1_t_l, bool_tf_0_1_f_l)


    bool_tf_0_1_b =   bool_tf_0_1*bool_tf_0_1_l

    bool_tf = bool_tf_0_1_b > 0.5

    image_np = tf.where(bool_tf, tf.ones([400,400]), tf.zeros([400,400]))
    image_np = tf.where(clip_tf, image_np, tf.ones([400,400]))

    image_np = tf.reshape(image_np,(400,400,1))
    image_np0 = tf.image.resize_images(image_np,(256,256))

    num_pix = tf.reduce_sum(image_np0)

    image_np2 = tf.reshape(image_np0,(256,256,1))
    image_np2_c = tf.image.crop_to_bounding_box(image_np2,154,96,64,64)
    bbox_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(output_bbox_map,1e-20,1e+20))*image_np2_c + tf.log(tf.clip_by_value(1.-output_bbox_map,1e-20,1e+20))*(tf.ones([64,64]) - image_np2_c) )

    row_idx1,col_idx1 = tf.meshgrid(tf.range(256),tf.range(256))
    result_x_p = tf.cast(-result_y * 256.0 / 80.0 + 4 * 256 / 5.0,tf.int32)
    result_y_p = tf.cast(-result_x * 256.0 / 80.0 + 256 / 2.0,tf.int32)
    d_p_x = tf.math.abs(result_y + (tf.cast(result_x_p,tf.float32) - tf.constant(4 * 256 / 5.0))*80.0/256.0 - y_conv[0][0])
    d_p_y = tf.math.abs(result_x + (tf.cast(result_y_p,tf.float32) - tf.constant( 256 / 2.0))*80.0/256.0 - y_conv[0][1])
    cond_p = tf.math.logical_and(tf.equal(row_idx1,result_y_p),tf.equal(col_idx1,result_x_p))
    p_map = tf.where(cond_p,tf.ones([256,256]), tf.zeros([256,256]))
    p_map_g = tf.exp(-(tf.cast(row_idx1-result_y_p,tf.float32)**2 + tf.cast(col_idx1-result_x_p,tf.float32)**2)/tf.constant(2.0*0.5))

    p_map = tf.reshape(p_map,(256,256,1))
    p_map_g = tf.reshape(p_map_g,(256,256,1))
    p_map_c = tf.image.crop_to_bounding_box(p_map,154,96,64,64)
    p_map_g_c = tf.image.crop_to_bounding_box(p_map_g,154,96,64,64)

    print (p_map_c)

    p_loss = -tf.reduce_sum(tf.log(tf.clip_by_value(output_pose_map,1e-20,1e+20))*p_map_c*(1.-output_pose_map)+tf.log(tf.clip_by_value(1.-output_pose_map,1e-20,1e+20))*(1.-p_map_c)*(1.-p_map_g_c)*output_pose_map)
    #p_loss = -tf.reduce_sum(p_map_g*output_pose_map)

    reg_loss = (d_p_x + d_p_y)/2.0
    yaw_loss = tf.math.abs(y_r[0][2] - y_conv[0][2])
    acc_p = tf.reduce_sum(output_pose_map*p_map_c)

    input_wp = tf.reshape(image1[0,2,:,:,0],(256,256))

    input_wp1 = tf.where(input_wp > 0.1,tf.zeros([256,256]),tf.ones([256,256]))
    input_map = tf.reshape(image1[0,1,:,:,0],(256,256,1))
    input_obj = tf.reshape(OGM_y[0,9,:,:,0],(256,256,1))
    input_wp1 = tf.reshape(input_wp1,(256,256,1))
    input_wp1_c = tf.image.crop_to_bounding_box(input_wp1,154,96,64,64)
    input_obj_c = tf.image.crop_to_bounding_box(input_obj,154,96,64,64)
    input_map_c = tf.image.crop_to_bounding_box(input_map,154,96,64,64)

    #num_p = tf.reduce_sum(tf.cast(tf.math.argmax(output_bbox_map,axis=0),tf.float32))
    wp_loss = tf.reduce_mean(tf.cast(output_bbox_map,tf.float32)*input_wp1_c)
    map_loss = tf.reduce_mean(tf.cast(output_bbox_map,tf.float32) * input_map_c)
    obj_loss = tf.reduce_mean(tf.cast(output_bbox_map,tf.float32)*input_obj_c)

    ssim_loss = tf.reduce_mean(tf.image.ssim(OGM_y, OGM_pred1, 1.0))
    gen_loss = - tf.reduce_mean(OGM_y * tf.log( tf.clip_by_value(OGM_pred,1e-20,1e+20)) + (1.-OGM_y) * tf.log( tf.clip_by_value(1.-OGM_pred,1e-20,1e+20)))


    pred_m = tf.reshape(output_pose_map,(64,64))
    pix_x = tf.cast(tf.math.argmax(tf.reduce_max(pred_m,reduction_indices=1)),tf.float32)+154.0
    pix_y = tf.cast(tf.math.argmax(tf.reduce_max(pred_m,reduction_indices=0)),tf.float32)+96.0
    pred_x = -(pix_x - 4.0*256.0/5.0)*80.0/256.0 + y_conv[0][0]
    pred_y = -(pix_y - 128.0)*80.0/256.0 + y_conv[0][1]
    loss_x = tf.math.abs(y_r[0][0] - pred_x)
    loss_y = tf.math.abs(y_r[0][1] - pred_y)
    pred_loss = tf.math.sqrt(loss_x**2 + loss_y**2) 

    #print p_map
    #print output_pose_map
    #print pred_loss

    ego_loss =  p_loss + reg_loss + yaw_loss  + gen_loss + bbox_loss
    env_loss = wp_loss + obj_loss + map_loss
    w_im = tf.random_uniform([1],minval=0,maxval=1,dtype=tf.float32)
    w_env = tf.random_uniform([1],minval=0,maxval=1,dtype=tf.float32)
    total_loss = ego_loss + env_loss #imitation drop-out
    train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss)
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(
        per_process_gpu_memory_fraction=0.8
    ))

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1.0)
    sess = tf.InteractiveSession(config=config)
    sess.run(tf.global_variables_initializer())

    filenames = []
    for num in range(120):
        filenames.append("scene"+str(num).zfill(5))

    train_image_list,train_r_label = make_datalist(filenames)

    filenames = []
    for num in range(120,150):
        filenames.append("scene"+str(num).zfill(5))
    valid_image_list,valid_r_label = make_datalist(filenames)

    filenames = []
    for num in range(150,180):
        filenames.append("scene"+str(num).zfill(5))
    test_image_list,test_r_label = make_datalist(filenames)
    print ("finish making dataset list")

    train_generator = GeneratorBuilder(train_image_list, train_r_label, batchSize)
    valid_generator = GeneratorBuilder(valid_image_list, valid_r_label, batchSize)
    test_generator = GeneratorBuilder(test_image_list, test_r_label, batchSize)

    train_new_generator = BackgroundGenerator(
        train_generator.build(), max_prefetch=100)
    valid_new_generator = BackgroundGenerator(
        valid_generator.build(), max_prefetch=100)
    test_new_generator = BackgroundGenerator(
        test_generator.build(), max_prefetch=100)
    trainrmse = 5000
    while(l <= 2000):
        print("start training")
        loss_a = 0
        loss_b = 0
        loss_c = 0
        loss_d = 0
        loss_e = 0
        loss_f = 0
        loss_g = 0
        loss_h = 0
        loss_j = 0
        for i, train_data in enumerate(train_new_generator):
            if not train_data:
                break
            else:
                #sess.run(tf.local_variables_initializer())
                # print("Next!")
                train_image = train_data[0]
                train_r = train_data[1]
                train_image_result = copy.copy(train_image)
                train_image_result[:,5,:,:,:] = np.zeros((1,256,256,1))
                train_image_result[:,6,:,:,:] = np.zeros((1,256,256,1))
                train_image_result[:,7,:,:,:] = np.zeros((1,256,256,1))
                train_image_result[:,8,:,:,:] = np.zeros((1,256,256,1))
                train_image_result[:,9,:,:,:] = np.zeros((1,256,256,1))
                loss_tmp = sess.run([train_step,ssim_loss,reg_loss,gen_loss,wp_loss,p_loss,obj_loss,yaw_loss,pred_loss,bbox_loss,result_x_p,result_y_p,pix_x,pix_y,acc_p,image_box_c,p_map_c,image_np2_c,input_wp1_c], feed_dict={
                         OGM_x:train_image_result[:,0:10,:,:,:],OGM_y:train_image[:,0:10,:,:,:],image1: train_image[:,10:13,:,:,:], y_r: train_r, keep_prob: 0.5})
                loss_a += loss_tmp[1]
                loss_b += loss_tmp[2]
                loss_c += loss_tmp[3]
                loss_d += loss_tmp[4]
                loss_e += loss_tmp[5]
                loss_f += loss_tmp[6]
                loss_g += loss_tmp[7]
                loss_h += loss_tmp[8]
                loss_j += loss_tmp[9]
                #if(i != 0):
                    #print (i,loss_tmp[1],loss_tmp[2],loss_tmp[3],loss_tmp[4],loss_tmp[5],loss_tmp[6],loss_tmp[7],[loss_tmp[8]],loss_tmp[9],loss_tmp[10],loss_tmp[11],loss_tmp[12],loss_tmp[13])
                    #print loss_tmp[16]*256/400
                    #cv2.imwrite("p_map.jpg",loss_tmp[16]*255)
                    #cv2.imwrite("box_map.jpg",255.-loss_tmp[17]*255)
                    #cv2.imwrite("input_wp.jpg",loss_tmp[18]*255)
                if(i % 1000 == 0 and i != 0):
                    print("train_step: " + str(i) +", ssim: "+str(loss_a/10)+", reg_loss: "+str(loss_b/1000)+", gen_loss: "+str(loss_c/1000)+", wp_loss: "+str(loss_d/1000)+", p_loss: "+str(loss_e/1000)+", obj_loss: "+str(loss_f/1000)+", yaw_loss: "+str(loss_g/1000)+", pred_loss: "+str(loss_h/1000)+", bbox_loss: "+str(loss_j/1000))

                    loss_a = 0
                    loss_b = 0
                    loss_c = 0
                    loss_d = 0
                    loss_e = 0
                    loss_f = 0
                    loss_g = 0
                    loss_h = 0
        if((l + 1) % 1 == 0):
            rmse = 0
            valid_rmse = 0
            num = 0
            valid_num = 0

            print("start valid data test")
            for i, valid_data in enumerate(valid_new_generator):
                if not valid_data:
                    break
                else:
                    valid_image = valid_data[0]
                    valid_r = valid_data[1]
                    valid_image_result = copy.copy(valid_image)
                    valid_image_result[:,5,:,:,:] = np.zeros((1,256,256,1))
                    valid_image_result[:,6,:,:,:] = np.zeros((1,256,256,1))
                    valid_image_result[:,7,:,:,:] = np.zeros((1,256,256,1))
                    valid_image_result[:,8,:,:,:] = np.zeros((1,256,256,1))
                    valid_image_result[:,9,:,:,:] = np.zeros((1,256,256,1))
                    result = sess.run([pred_loss,y_conv], feed_dict={OGM_x:valid_image_result[:,0:10,:,:,:],OGM_y:valid_image[:,0:10,:,:,:],image1: valid_image[:,10:13,:,:,:], y_r: valid_r, keep_prob: 1.0})
                    valid_rmse = valid_rmse + result[0]
                    #print(result[1])
                    valid_num += 1
            valid_rmse = valid_rmse / valid_num

            print("start test data test")
            for i, test_data in enumerate(test_new_generator):
                if not test_data:
                    break
                else:
                    test_image = test_data[0]
                    test_r = test_data[1]
                    test_image_result = copy.copy(test_image)
                    test_image_result[:,5,:,:,:] = np.zeros((1,256,256,1))
                    test_image_result[:,6,:,:,:] = np.zeros((1,256,256,1))
                    test_image_result[:,7,:,:,:] = np.zeros((1,256,256,1))
                    test_image_result[:,8,:,:,:] = np.zeros((1,256,256,1))
                    test_image_result[:,9,:,:,:] = np.zeros((1,256,256,1))
                    result = sess.run([pred_loss,y_conv], feed_dict={OGM_x:test_image_result[:,0:10,:,:,:],OGM_y:test_image[:,0:10,:,:,:],image1: test_image[:,10:13,:,:,:], y_r: test_r, keep_prob: 1.0})
                    rmse = rmse + result[0]
                    #print(result[1])
                        
                    num += 1
            rmse = rmse / num

            f = open("loss/model_ours_true_anno.txt", "a")

            if(trainrmse > valid_rmse):
                trainrmse = valid_rmse
                save_path = saver.save(sess, "model/ours"+str(l)+"_true_anno.ckpt")
                f.write("epoch:=%d , train_rmse:=%f, valid_rmse:=%f, test_rmse:=%f ModelSave!\n" % (
                    l, trainrmse, valid_rmse, rmse))
                print("epoch:=%d , train_rmse:=%f, valid_rmse:=%f ,test_rmse:=%f ModelSave!\n" %
                      (l, trainrmse, valid_rmse, rmse))
            else:
                f.write("epoch:=%d ,train_rmse:=%f, valid_rmse:=%f, test_rmse:=%f\n" %
                        (l, trainrmse, valid_rmse, rmse))
                print("epoch:=%d ,train_rmse:=%f, valid_rmse:=%f, test_rmse:=%f\n" %
                      (l, trainrmse, valid_rmse, rmse))
            f.close()
        l += 1
