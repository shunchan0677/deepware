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
                image = np.zeros((32,400,400,1))
                for flame in range(self.num_flame):
                    path = self.image_label[self.random_seq[l]]+"/bbox_"+str(flame)+".png"
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
                        image[flame,:,:,:] = np.asarray(cv2.imread(self.image_label[self.random_seq[l]]+"/bbox_"+str(flame)+".png",cv2.IMREAD_GRAYSCALE)).reshape((1,400,400,1))/ 255.0
                        image[flame+10,:,:,:] = np.asarray(cv2.imread(self.image_label[self.random_seq[l]]+"/ego_pose_"+str(flame)+".png",cv2.IMREAD_GRAYSCALE)).reshape((1,400,400,1))/ 255.0
                        x = self.r_label[self.random_seq[l]][flame*3+1]
                        y = self.r_label[self.random_seq[l]][flame*3]
                        x_im = -y * 400.0 / 80.0 + 4 * 400 / 5.0
                        y_im = -x *  400 / 80.0 + 400 / 2.0
                        image[flame+20, int(x_im) , int(y_im) ,0] = 1.0
                    image[30,:,:,:] = np.asarray(cv2.imread(self.image_label[self.random_seq[l]]+"/map.png",cv2.IMREAD_GRAYSCALE)).reshape((1,400,400,1))/ 255.0
                    image[31,:,:,:] = np.asarray(cv2.imread(self.image_label[self.random_seq[l]]+"/waypoint.png",cv2.IMREAD_GRAYSCALE)).reshape((1,400,400,1))/ 255.0
                    self.train_image=[image]
                    for i in range(10):
                        self.train_r.append(self.r_label[self.random_seq[l]][i*3:i*3+3])
                    if(True):
                        self.train_image = np.asarray(self.train_image).reshape((1,32,400,400,1))
                        yield [self.train_image, self.train_r]
                        self.train_image = []
                        self.train_r = []
            yield None


def make_datalist(filenames):
    train_image_label = []
    train_r_label = []

    for filename in filenames:
        image_file = glob.glob("/home/s_seiya/tmp/lyft_data_big2/" +
                               filename + "/*")
        image_file.sort()
        train_image_label+=image_file

        r = np.loadtxt("/home/s_seiya/tmp/lyft_data_big2/" + filename +
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


if __name__ == '__main__':
    print("Ours training result log")
    tf.reset_default_graph()
    l = 0
    basermse = 1000
    batchSize = 1

    OGM_x = tf.placeholder(tf.float32, [1,10] + [400, 400]+ [1])
    ego_bbox_x = tf.placeholder(tf.float32, [1,10] + [400, 400]+ [1])
    ego_p_x = tf.placeholder(tf.float32, [1,10] + [400, 400]+ [1])
    maps_x = tf.placeholder(tf.float32, [1,2] + [400, 400]+ [1])

    OGM_y1 = tf.placeholder(tf.float32, [1,10] + [400, 400]+ [1])
    ego_bbox_y = tf.placeholder(tf.float32, [1,10] + [400, 400]+ [1])
    ego_p_y = tf.placeholder(tf.float32, [1,10] + [400, 400]+ [1])
    y_r = tf.placeholder(tf.float32, [10,3])
    keep_prob = tf.placeholder(tf.float32)

    OGM_pred = OGMPred(OGM_x)
    OGM_pred1 = tf.where(OGM_pred > 0.1,tf.ones([10,400,400,1]),tf.zeros([10,400,400,1]))
    OGM_y = tf.where(OGM_y1 > 0.1,tf.ones([1,10,400,400,1]),tf.zeros([1,10,400,400,1]))
    y_conv,output_bbox_map,output_pose_map = AgentNet(OGM_pred1, ego_bbox_x, ego_p_x, maps_x, keep_prob)

    result_x = y_r[:,1]#result[0]
    result_y = y_r[:,0]#result[1]

    yaw = y_r[:,2]#tf.math.atan(result_x/(r-result_y))#y_r[0][2] #
    image_box_c = ego_bbox_x
    image_np2_c = tf.reshape(ego_bbox_y,[10, 400,400,1])

    bbox_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(output_bbox_map,1e-20,1e+20))*image_np2_c + tf.log(tf.clip_by_value(1.-output_bbox_map,1e-20,1e+20))*(1. - image_np2_c) )

    down_factor = 1

    #row_idx1,col_idx1 = tf.meshgrid(tf.range(400/down_factor),tf.range(400/down_factor))
    result_x_p = tf.cast(-result_y * (400.0 /down_factor) / 80.0 + 4 * (400 /down_factor) / 5.0,tf.int32)
    result_y_p = tf.cast(-result_x * (400.0 /down_factor) / 80.0 + (400 /down_factor) / 2.0,tf.int32)
    print(result_x_p)
    d_p_x = tf.math.abs(result_y + (tf.cast(result_x_p,tf.float32) - tf.constant(4 * (400 /down_factor) / 5.0))*80.0/(400.0/down_factor) - y_conv[:,0])
    d_p_y = tf.math.abs(result_x + (tf.cast(result_y_p,tf.float32) - tf.constant( (400/down_factor) / 2.0))*80.0/(400.0/down_factor) - y_conv[:,1])
    #p_map_g = tf.exp(-(tf.cast(row_idx1-result_y_p,tf.float32)**2 + tf.cast(col_idx1-result_x_p,tf.float32)**2)/tf.constant(2.0*0.5))

    p_map_c = ego_p_y
    #p_map_g_c = p_map_g

    #p_loss = -tf.reduce_sum(tf.log(tf.clip_by_value(output_pose_map,1e-20,1e+20))*p_map_c*(1.-output_pose_map)**2+tf.log(tf.clip_by_value(1.-output_pose_map,1e-20,1e+20))*(1.-p_map_c)*(1.-p_map_g_c)**4*output_pose_map**2)
    p_loss = -tf.reduce_sum(tf.log(tf.clip_by_value(output_pose_map,1e-20,1e+20))*p_map_c)#*(1.-output_pose_map) + tf.log(tf.clip_by_value(1.-output_pose_map,1e-20,1e+20))*(1.-p_map_c)*output_pose_map)

    reg_loss = tf.reduce_mean(tf.math.sqrt(d_p_x*2 + d_p_y*2))
    yaw_loss = tf.reduce_sum(tf.math.abs(y_r[:,2] - y_conv[:,2]))
    acc_p = tf.reduce_sum(output_pose_map*p_map_c)

    input_wp = tf.reshape(maps_x[0,1,:,:,0],(400,400))

    input_wp1 = tf.where(input_wp > 0.1,tf.zeros([400,400]),tf.ones([400,400]))
    input_map = tf.reshape(maps_x[0,0,:,:,0],(400,400,1))
    input_obj = tf.reshape(OGM_y[0,9,:,:,0],(400,400,1))
    input_wp1 = tf.reshape(input_wp1,(400,400,1))
    input_wp1_c = tf.image.resize_images(input_wp1,(400/down_factor,400/down_factor))#tf.image.crop_to_bounding_box(input_wp1,240,150,100,100)
    input_obj_c = tf.image.resize_images(input_obj,(400/down_factor,400/down_factor))#tf.image.crop_to_bounding_box(input_obj,240,150,100,100)
    input_map_c = tf.image.resize_images(input_map,(400/down_factor,400/down_factor))#tf.image.crop_to_bounding_box(input_map,240,150,100,100)

    #num_p = tf.reduce_sum(tf.cast(tf.math.argmax(output_bbox_map,axis=0),tf.float32))
    wp_loss = tf.reduce_mean(tf.cast(output_bbox_map,tf.float32)*input_wp1_c)
    map_loss = tf.reduce_mean(tf.cast(output_bbox_map,tf.float32) * input_map_c)
    obj_loss = tf.reduce_mean(tf.cast(output_bbox_map,tf.float32)*input_obj_c)

    ssim_loss = tf.reduce_mean(tf.image.ssim(OGM_y, OGM_pred1, 1.0))
    gen_loss = - tf.reduce_mean(OGM_y * tf.log( tf.clip_by_value(OGM_pred,1e-20,1e+20)) + (1.-OGM_y) * tf.log( tf.clip_by_value(1.-OGM_pred,1e-20,1e+20)))


    pred_m = tf.reshape(output_pose_map[0,9,:,:,0],(400/down_factor,400/down_factor))
    pix_x = tf.cast(tf.math.argmax(tf.reduce_max(pred_m,reduction_indices=1)),tf.float32)
    pix_y = tf.cast(tf.math.argmax(tf.reduce_max(pred_m,reduction_indices=0)),tf.float32)
    pred_x = -(pix_x - 4.0*(400/down_factor)/5.0)*80.0/(400/down_factor) + y_conv[9,0]
    pred_y = -(pix_y - (400/down_factor)/2.0)*80.0/(400/down_factor) + y_conv[9,1]
    loss_x = tf.math.abs(y_r[9,0] - pred_x)
    loss_y = tf.math.abs(y_r[9,1] - pred_y)
    pred_loss = tf.math.sqrt(loss_x**2 + loss_y**2)

    simple_loss = tf.math.sqrt(tf.math.abs(y_r[:,0] - y_conv[:,0])**2 + tf.math.abs(y_r[:,1] - y_conv[:,1])**2)
    pred_loss = simple_loss[9]


    ego_loss = gen_loss  + simple_loss
    env_loss = wp_loss + obj_loss + map_loss
    w_im = tf.random_uniform([1],minval=0,maxval=1,dtype=tf.float32)
    w_env = tf.random_uniform([1],minval=0,maxval=1,dtype=tf.float32)
    total_loss = ego_loss #+ env_loss #imitation drop-out
    train_step = tf.train.AdamOptimizer(1e-5).minimize(total_loss)
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
                train_image_result[:,5,:,:,:] = np.zeros((1,400,400,1))
                train_image_result[:,6,:,:,:] = np.zeros((1,400,400,1))
                train_image_result[:,7,:,:,:] = np.zeros((1,400,400,1))
                train_image_result[:,8,:,:,:] = np.zeros((1,400,400,1))
                train_image_result[:,9,:,:,:] = np.zeros((1,400,400,1))

                train_image_result[:,15,:,:,:] = np.zeros((1,400,400,1))
                train_image_result[:,16,:,:,:] = np.zeros((1,400,400,1))
                train_image_result[:,17,:,:,:] = np.zeros((1,400,400,1))
                train_image_result[:,18,:,:,:] = np.zeros((1,400,400,1))
                train_image_result[:,19,:,:,:] = np.zeros((1,400,400,1))

                train_image_result[:,25,:,:,:] = np.zeros((1,400,400,1))
                train_image_result[:,26,:,:,:] = np.zeros((1,400,400,1))
                train_image_result[:,27,:,:,:] = np.zeros((1,400,400,1))
                train_image_result[:,28,:,:,:] = np.zeros((1,400,400,1))
                train_image_result[:,29,:,:,:] = np.zeros((1,400,400,1))


                loss_tmp = sess.run([train_step,ssim_loss,reg_loss,gen_loss,wp_loss,p_loss,obj_loss,yaw_loss,pred_loss,bbox_loss], feed_dict={
                         OGM_x:train_image_result[:,0:10,:,:,:],OGM_y:train_image[:,0:10,:,:,:],ego_bbox_x:train_image_result[:,10:20,:,:,:],ego_bbox_y:train_image[:,10:20,:,:,:],ego_p_x:train_image_result[:,20:30,:,:,:],ego_p_y:train_image[:,20:30,:,:,:],maps_x:train_image[:,30:,:,:,:], y_r: train_r, keep_prob: 0.5})
                loss_a += loss_tmp[1]
                loss_b += loss_tmp[2]
                loss_c += loss_tmp[3]
                loss_d += loss_tmp[4]
                loss_e += loss_tmp[5]
                loss_f += loss_tmp[6]
                loss_g += loss_tmp[7]
                loss_h += loss_tmp[8]
                loss_j += loss_tmp[9]
                if(i % 100 == 0 and i != 0):
                    print("train_step: " + str(i) +", ssim: "+str(loss_a/1)+", reg_loss: "+str(loss_b/100)+", gen_loss: "+str(loss_c/100)+", wp_loss: "+str(loss_d/100)+", p_loss: "+str(loss_e/100)+", obj_loss: "+str(loss_f/100)+", yaw_loss: "+str(loss_g/100)+", pred_loss: "+str(loss_h/100)+", bbox_loss: "+str(loss_j/100))

                    loss_a = 0
                    loss_b = 0
                    loss_c = 0
                    loss_d = 0
                    loss_e = 0
                    loss_f = 0
                    loss_g = 0
                    loss_h = 0
                    loss_j = 0
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
                    valid_image_result[:,5,:,:,:] = np.zeros((1,400,400,1))
                    valid_image_result[:,6,:,:,:] = np.zeros((1,400,400,1))
                    valid_image_result[:,7,:,:,:] = np.zeros((1,400,400,1))
                    valid_image_result[:,8,:,:,:] = np.zeros((1,400,400,1))
                    valid_image_result[:,9,:,:,:] = np.zeros((1,400,400,1))

                    valid_image_result[:,15,:,:,:] = np.zeros((1,400,400,1))
                    valid_image_result[:,16,:,:,:] = np.zeros((1,400,400,1))
                    valid_image_result[:,17,:,:,:] = np.zeros((1,400,400,1))
                    valid_image_result[:,18,:,:,:] = np.zeros((1,400,400,1))
                    valid_image_result[:,19,:,:,:] = np.zeros((1,400,400,1))

                    valid_image_result[:,25,:,:,:] = np.zeros((1,400,400,1))
                    valid_image_result[:,26,:,:,:] = np.zeros((1,400,400,1))
                    valid_image_result[:,27,:,:,:] = np.zeros((1,400,400,1))
                    valid_image_result[:,28,:,:,:] = np.zeros((1,400,400,1))
                    valid_image_result[:,29,:,:,:] = np.zeros((1,400,400,1))
                    result = sess.run([pred_loss,y_conv], feed_dict={
OGM_x:valid_image_result[:,0:10,:,:,:],OGM_y:valid_image[:,0:10,:,:,:],ego_bbox_x:valid_image_result[:,10:20,:,:,:],ego_bbox_y:valid_image[:,10:20,:,:,:],ego_p_x:valid_image_result[:,20:30,:,:,:],ego_p_y:valid_image[:,20:30,:,:,:],maps_x:valid_image[:,30:,:,:,:], y_r: valid_r, keep_prob: 1.0})
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
                    test_image_result[:,5,:,:,:] = np.zeros((1,400,400,1))
                    test_image_result[:,6,:,:,:] = np.zeros((1,400,400,1))
                    test_image_result[:,7,:,:,:] = np.zeros((1,400,400,1))
                    test_image_result[:,8,:,:,:] = np.zeros((1,400,400,1))
                    test_image_result[:,9,:,:,:] = np.zeros((1,400,400,1))

                    test_image_result[:,15,:,:,:] = np.zeros((1,400,400,1))
                    test_image_result[:,16,:,:,:] = np.zeros((1,400,400,1))
                    test_image_result[:,17,:,:,:] = np.zeros((1,400,400,1))
                    test_image_result[:,18,:,:,:] = np.zeros((1,400,400,1))
                    test_image_result[:,19,:,:,:] = np.zeros((1,400,400,1))

                    test_image_result[:,25,:,:,:] = np.zeros((1,400,400,1))
                    test_image_result[:,26,:,:,:] = np.zeros((1,400,400,1))
                    test_image_result[:,27,:,:,:] = np.zeros((1,400,400,1))
                    test_image_result[:,28,:,:,:] = np.zeros((1,400,400,1))
                    test_image_result[:,29,:,:,:] = np.zeros((1,400,400,1))

                    result = sess.run([pred_loss,y_conv], feed_dict={OGM_x:test_image_result[:,0:10,:,:,:],OGM_y:test_image[:,0:10,:,:,:],ego_bbox_x:test_image_result[:,10:20,:,:,:],ego_bbox_y:test_image[:,10:20,:,:,:],ego_p_x:test_image_result[:,20:30,:,:,:],ego_p_y:test_image[:,20:30,:,:,:],maps_x:test_image[:,30:,:,:,:], y_r: test_r, keep_prob: 1.0})
                    rmse = rmse + result[0]
                    #print(result[1])

                    num += 1
            rmse = rmse / num

            f = open("loss/model_ae_sim_iv2.txt", "a")

            if(trainrmse > valid_rmse):
                trainrmse = valid_rmse
                save_path = saver.save(sess, "model/ae_sim_iv"+str(l)+".ckpt")
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
