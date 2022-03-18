#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np
import random
import glob
import threading
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
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
                    path = self.image_label[self.random_seq[l]]+"/occupancy_grid_"+str(flame)+"_.jpg"
                    if(os.path.exists(path) == False):
                        flag = False
                path = self.image_label[self.random_seq[l]]+"/_vector_image_raw_ego_vehicle_.jpg"
                if(os.path.exists(path) == False):
                    flag = False
                path = self.image_label[self.random_seq[l]]+"/_vector_image_raw_hd_map_.jpg"
                if(os.path.exists(path) == False):
                    flag = False
                path = self.image_label[self.random_seq[l]]+"/_vector_image_raw_waypoint_.jpg"
                if(os.path.exists(path) == False):
                    flag = False

                if(flag):
                    for flame in range(self.num_flame):
                        image[flame,:,:,:] = np.asarray(cv2.imread(self.image_label[self.random_seq[l]]+"/occupancy_grid_"+str(flame)+"_.jpg",cv2.IMREAD_GRAYSCALE)).reshape((1,256,256,1))/ 255.0
                    image[10,:,:,:] = np.asarray(cv2.imread(self.image_label[self.random_seq[l]]+"/_vector_image_raw_ego_vehicle_.jpg",cv2.IMREAD_GRAYSCALE)).reshape((1,256,256,1))/ 255.0
                    image[11,:,:,:] = np.asarray(cv2.imread(self.image_label[self.random_seq[l]]+"/_vector_image_raw_hd_map_.jpg",cv2.IMREAD_GRAYSCALE)).reshape((1,256,256,1))/ 255.0
                    image[12,:,:,:] = np.asarray(cv2.imread(self.image_label[self.random_seq[l]]+"/_vector_image_raw_waypoint_.jpg",cv2.IMREAD_GRAYSCALE)).reshape((1,256,256,1))/ 255.0
                    self.train_image.append(image)
                    self.train_r.append(self.r_label[self.random_seq[l],12:14])
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
        image_file = glob.glob("/media/brainiv/PioMeidai/CIA_dataset/good/withFM/" +
                               filename + "/images/*")
        image_file.sort()
        train_image_label+=image_file

        r = np.loadtxt("/media/brainiv/PioMeidai/CIA_dataset/good/withFM/" + filename +
                       "/tf.csv", delimiter=",")
        train_r_label = r

    return train_image_label, train_r_label

def condition_i(i, j, min_x, max_x, min_y, max_y, image):
    return i < max_x

def condition_j(i, j, min_x, max_x, min_y, max_y, image):
    return j < max_y

def update_x(i, j, min_x, max_x, min_y, max_y, image):
    init_val2 = (i, j, min_x, max_x, min_y, max_y, image)
    i, j, min_x, max_x, min_y, max_y, image = loop2 = tf.while_loop(condition_j,update_y,init_val2)
    return i+1, j, min_x, max_x, min_y, max_y, image


def update_y(i, j, min_x, max_x, min_y, max_y, image):
    print image
    images = tf.Variable(np.zeros([400,400,1],np.int32))
    images.assign(image)
    images = images[i,j,0].assign(1)
    return i, j+1, min_x, max_x, min_y, max_y, images




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
    #inputs = tf.where(inputs > 0.1,tf.ones([13,256,256,1]),tf.zeros([13,256,256,1]))
    #inputs = tf.reshape(inputs,[1,13]+[256,256]+[1])
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

    #h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    with tf.name_scope('fc3') as scope:
        W_fc3 = weight_variable([100, 4])
        b_fc3 = bias_variable([4])
        y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3

    return y_conv


if __name__ == '__main__':
    tf.reset_default_graph()
    l = 0
    basermse = 1000
    batchSize = 1

    OGM_x = tf.placeholder(tf.float32, [None,10] + [256, 256]+ [1])
    image1 = tf.placeholder(tf.float32, [None,3] + [256, 256]+ [1])
    OGM_y = tf.placeholder(tf.float32, [None,10] + [256, 256]+ [1])
    y_r = tf.placeholder(tf.float32, [None,2])
    keep_prob = tf.placeholder(tf.float32)

    OGM_pred = OGMPred(OGM_x)
    OGM_pred1 = tf.where(OGM_pred > 0.1,tf.ones([10,256,256,1]),tf.zeros([10,256,256,1]))
    y_conv = AgentNet(OGM_pred1, image1, keep_prob)
    mean_x = y_conv[0][0]
    var_x = tf.math.abs(y_conv[0][1])
    mean_y = y_conv[0][2]
    var_y = tf.math.abs(y_conv[0][3])

    #p_x = tfd.Normal(loc=mean_x,scale=var_x)
    #p_y = tfd.Normal(loc=mean_y,scale=var_y)
    p = tfd.MultivariateNormalDiag(loc=[mean_x,mean_y], scale_diag=[var_x,var_y])

    #result_x = p_x.sample()
    #result_y = p_y.sample()
 
    #likelihood_x = - tf.reduce_sum(p_x.log_prob(y_r[0][0]))
    #likelihood_y = - tf.reduce_sum(p_y.log_prob(y_r[0][1]))

    result = p.sample()
    result_x = result[0]
    result_y = result[1]

    loss_x = tf.math.abs(y_r[0][0] - result_x)
    loss_y = tf.math.abs(y_r[0][1] - result_y)

    reg_loss = loss_x + loss_y

    #reg_loss_like = likelihood_x + likelihood_y
    l2 = result_x**2 + result_y**2
    r = l2 / (2*result_y)
    yaw = tf.math.atan(result_x/(r-result_y))#tf.constant(0,tf.float32)

    image_center_point = tf.constant([[400 / 2, int(4 * 400 / 5.0)],[400 / 2, int(4 * 400 / 5.0)],[400 / 2, int(4 * 400 / 5.0)],[400 / 2, int(4 * 400 / 5.0)]])

    box_position_velo = [result_x, result_y]

    rot_mat = [[tf.math.cos(yaw), -tf.math.sin(yaw)], [tf.math.sin(yaw), tf.math.cos(yaw)]]
    box_size = tf.constant([1.695 / 2.0, 4.36 / 2.0])
    box_mat = tf.constant([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])
    box_4 = box_size * box_mat
    rot_res = tf.matmul(box_4, rot_mat)

    rot_res_rated = (box_position_velo + rot_res) * 400 / 60.0
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
    beta2 = tf.where(condition, tf.cast(image_box_c[3,1],tf.float32) - alpha*tf.cast(image_box_c[3,0],tf.float32), tf.cast(image_box_c[3,0],tf.float32))
    bool_mat_0_1 = tf.where(condition, tf.cast(col_idx,tf.float32)-tf.cast(row_idx,tf.float32)*alpha, tf.cast(row_idx,tf.float32))
    map_1 = tf.where(bool_mat_0_1 > beta,tf.ones([400,400]), tf.zeros([400,400]))
    map_2 = tf.where(bool_mat_0_1 < beta2,tf.ones([400,400]), tf.zeros([400,400]))
    bool_tf_0_1_t = (map_1*map_2)
    map_1 = tf.where(bool_mat_0_1 < beta,tf.ones([400,400]), tf.zeros([400,400]))
    map_2 = tf.where(bool_mat_0_1 > beta2,tf.ones([400,400]), tf.zeros([400,400]))
    bool_tf_0_1_f = (map_1*map_2)
    condition2 = tf.cast(alpha < tf.constant(0.0),tf.bool)
    bool_tf_0_1 = tf.where(condition2, bool_tf_0_1_t, bool_tf_0_1_f)
    condition3_0 = tf.cast(alpha == tf.constant(0.0),tf.bool)
    condition3_1 = tf.cast(d_y1 > tf.constant(0.0),tf.bool)
    condition3 = tf.math.logical_and(condition3_0,condition3_1)
    bool_tf_0_1 = tf.where(condition3, bool_tf_0_1_f, bool_tf_0_1)

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
    condition2_1_l = tf.cast(d_y1_l < tf.constant(0.0),tf.bool)
    condition2_l = tf.math.logical_and(condition2_0_l,condition2_1_l)
    bool_tf_0_1_l = tf.where(condition2_l, bool_tf_0_1_t_l, bool_tf_0_1_f_l)
    #condition3_0_l = tf.cast(alpha_l == tf.constant(0.0),tf.bool)
    #condition3_1_l = tf.cast(d_y1_l < tf.constant(0.0),tf.bool)
    #condition3_l = tf.math.logical_and(condition3_0_l,condition3_1_l)
    #bool_tf_0_1_l = tf.where(condition3_l, bool_tf_0_1_f_l, bool_tf_0_1_l)


    bool_tf_0_1_b =  bool_tf_0_1_l * bool_tf_0_1

    bool_tf = bool_tf_0_1_b > 0.5

    image_np = tf.where(bool_tf, tf.ones([400,400]), tf.zeros([400,400]))
    image_np = tf.where(clip_tf, image_np, tf.ones([400,400]))

    image_np = tf.reshape(image_np,(400,400,1))
    image_np0 = tf.image.resize_images(image_np,(256,256))

    num_pix = tf.reduce_sum(image_np0)

    image_np2 = tf.reshape(image_np0,(256,256))
    input_wp = tf.reshape(image1[0,2,:,:,0],(256,256))

    input_wp1 = tf.where(input_wp > 0.1,tf.zeros([256,256]),tf.ones([256,256]))
    input_map = tf.reshape(image1[0,1,:,:,0],(256,256))
    input_obj = tf.reshape(OGM_y[0,9,:,:,0],(256,256))

    wp_loss =  tf.reduce_sum(image_np2 * input_wp1) / num_pix
    map_loss = tf.reduce_sum(image_np2 * input_map) / num_pix
    obj_loss = tf.reduce_sum(image_np2 * input_obj) / num_pix

    ssim_loss = tf.reduce_mean(tf.image.ssim(OGM_y, OGM_pred1, 1.0))
    gen_loss = - tf.reduce_mean(OGM_y * tf.log( tf.clip_by_value(OGM_pred,1e-20,1e+20)) + (1.-OGM_y) * tf.log( tf.clip_by_value(1.-OGM_pred,1e-20,1e+20)))

    total_loss = tf.reduce_mean(gen_loss+reg_loss+wp_loss+obj_loss)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss)
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(
        per_process_gpu_memory_fraction=0.8
    ))

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1.0)
    sess = tf.InteractiveSession(config=config)
    sess.run(tf.global_variables_initializer())

    filenames = ["2019-07-19-15-43-50"]

    train_image_list,train_r_label = make_datalist(filenames)

    filenames = ["2019-07-19-15-43-50"]
    test_image_list,test_r_label = make_datalist(filenames)
    print ("finish making dataset list")

    train_generator = GeneratorBuilder(train_image_list, train_r_label, batchSize)
    test_generator = GeneratorBuilder(test_image_list, test_r_label, batchSize)

    train_new_generator = BackgroundGenerator(
        train_generator.build(), max_prefetch=100)
    test_new_generator = BackgroundGenerator(
        test_generator.build(), max_prefetch=100)

    while(l <= 2000):
        print("start training")
        loss_a = 0
        loss_b = 0
        loss_c = 0
        loss_d = 0
        loss_e = 0
        loss_f = 0
        loss_g = 0
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
                loss_tmp = sess.run([train_step,ssim_loss,reg_loss,gen_loss,wp_loss,map_loss,obj_loss,num_pix,min_x,max_x,min_y,max_y,image_box_c,alpha_l,beta_l,beta2_l,d_x_l,d_y_l,condition_l,condition2_0_l], feed_dict={
                         OGM_x:train_image_result[:,0:10,:,:,:],OGM_y:train_image[:,0:10,:,:,:],image1: train_image[:,10:13,:,:,:], y_r: train_r, keep_prob: 0.5})
                loss_a += loss_tmp[1]
                loss_b += loss_tmp[2]
                loss_c += loss_tmp[3]
                loss_d += loss_tmp[4]
                loss_e += loss_tmp[5]
                loss_f += loss_tmp[6]
                loss_g += loss_tmp[7]
                print i,loss_tmp[4],loss_tmp[5],loss_tmp[6],loss_tmp[7],loss_tmp[8],loss_tmp[9],loss_tmp[10],loss_tmp[11],loss_tmp[13],loss_tmp[14],loss_tmp[15],loss_tmp[16],loss_tmp[17],loss_tmp[18],loss_tmp[19]
                print loss_tmp[12]
                if(i % 1000 == 0 and i != 0):
                    print("train_step: " + str(i) +", ssim: "+str(loss_a/10)+", reg_loss: "+str(loss_b/1000)+", gen_loss: "+str(loss_c/1000)+", wp_loss: "+str(loss_d/1000)+", map_loss: "+str(loss_e/1000)+", obj_loss: "+str(loss_f/1000)+", im_map: "+str(loss_g/1000))
                    loss_a = 0
                    loss_b = 0
                    loss_c = 0
                    loss_d = 0
                    loss_e = 0
                    loss_f = 0
                    loss_g = 0
        if((l + 1) % 1 == 0):
            trainrmse = 0
            rmse = 0
            num = 0

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
                    result = sess.run([gen_loss,y_conv], feed_dict={OGM_x:test_image_result[:,0:10,:,:,:],OGM_y:test_image[:,0:10,:,:,:],image1: test_image[:,10:13,:,:,:], y_r: test_r, keep_prob: 1.0})
                    rmse = rmse + result[0]
                    print(result[1])
                        
                    num += 1
            rmse = rmse / num

            f = open("loss/model_m0_11.txt", "a")

            if(True):
                basermse = rmse
                save_path = saver.save(sess, "model/m0_11_"+str(l)+".ckpt")
                f.write("epoch:=%d ,train_acc:=%f ,test_acc:=%f ModelSave!\n" % (
                    l, trainrmse, rmse))
                print("epoch:=%d ,train_acc:=%f ,test_acc:=%f ModelSave!\n" %
                      (l, trainrmse, rmse))
            else:
                f.write("epoch:=%d ,train_acc:=%f ,test_acc:=%f\n" %
                        (l, trainrmse, rmse))
                print("epoch:=%d ,train_acc:=%f ,test_acc:=%f\n" %
                      (l, trainrmse, rmse))
            f.close()
        l += 1
