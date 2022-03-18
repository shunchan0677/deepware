#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
import random
import glob
import threading
import sys
from cell import ConvLSTMCell
import copy

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
    def __init__(self, image_label, batch_size):
        self.train_image = []
        self.train_r = []
        self.image_label = image_label
        self.batch_size = batch_size
        self.num_flame = 10

    def build(self):
        while True:
            self.random_seq = range(len(self.image_label))
            #random.shuffle(self.random_seq)
            for l in range(len(self.image_label)):
                image = np.zeros((self.num_flame,256,256,1))
                for flame in range(self.num_flame):
                    image[flame,:,:,:] = np.asarray(cv2.imread(self.image_label[self.random_seq[l]]+"/occupancy_grid_"+str(flame)+"_.jpg",cv2.IMREAD_GRAYSCALE)).reshape((1,256,256,1))/ 255.0
                self.train_image.append(image)
                if((l + 1) % self.batch_size == 0):
                    self.train_image = np.asarray(self.train_image).reshape((-1,self.num_flame,256,256,1))
                    self.train_r = copy.copy(self.train_image)
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
        image_file = glob.glob("/media/brainiv/PioMeidai/CIA_dataset/" +
                               filename + "/images/*")
        image_file.sort()
        train_image_label=image_file[22000:22100]
    return train_image_label


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],
                        padding='SAME')

def deconv2d(x,W,size,n2,chan):
    return tf.nn.conv2d_transpose(x, W, tf.stack([chan, size, size, n2]), strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def CNN(cell1, input_state1, cell2, input_state2, cell3, input_state3, cell4, input_state4, image, keep_prob):
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

    h_pool2_ = tf.reshape(h_pool2,[-1,1]+[64,64]+[32])

    with tf.name_scope('core-RNN-conv1') as scope:
        outputs1, state1 = tf.nn.dynamic_rnn(cell=cell1, inputs=h_pool2_, dtype=h_pool2_.dtype, initial_state=input_state1)

    outputs_ = tf.reshape(outputs1,[-1]+[64,64]+[32])

    with tf.name_scope('dec-conv1') as scope:
        W_conv3 = weight_variable([3, 3, 16, 32])
        b_conv3 = bias_variable([16])
        h_conv3 = tf.nn.relu(deconv2d(outputs_, W_conv3,128,16,1) + b_conv3)

    with tf.name_scope('dec-conv2') as scope:
        W_conv4 = weight_variable([3, 3, 1, 16]) 
        b_conv4 = bias_variable([1])
        h_conv4 = tf.nn.tanh(deconv2d(h_conv3, W_conv4,256,1,1) + b_conv4)

    ok = h_conv4 + image


    with tf.name_scope('class-conv1') as scope:
        W_conv1_1 = weight_variable([3, 3, 1, 16])
        b_conv1_1 = bias_variable([16])
        h_conv1_1 = tf.nn.relu(conv2d(ok, W_conv1_1) + b_conv1_1)
        h_pool1_1 = max_pool_2x2(h_conv1_1)


    with tf.name_scope('class-conv2') as scope:
        W_conv1_2 = weight_variable([3, 3, 1, 16]) 
        b_conv1_2 = bias_variable([1])
        h_conv1_2 = tf.sigmoid(deconv2d(h_pool1_1, W_conv1_2,256,1,1) + b_conv1_2) 

    return h_conv1_2, state1,state1,state1,state1, h_conv4


if __name__ == '__main__':
    l = 0
    basermse = 1000
    batchSize = 1

    image1 = tf.placeholder(tf.float32, [None,1] + [256, 256]+ [1])
    y_r = tf.placeholder(tf.float32, [None,1] + [256, 256]+ [1])
    keep_prob = tf.placeholder(tf.float32)
    
    cell1 = ConvLSTMCell([64,64], 32, [3, 3],"1")
    input_state1 = cell1.zero_state(1,dtype=tf.float32)

    cell2 = ConvLSTMCell([64,64], 32, [3, 3],"2")
    input_state2 = cell2.zero_state(1,dtype=tf.float32)

    cell3 = ConvLSTMCell([64,64], 32, [3, 3],"3")
    input_state3 = cell3.zero_state(1,dtype=tf.float32)

    cell4 = ConvLSTMCell([64,64], 32, [3, 3],"4")
    input_state4 = cell4.zero_state(1,dtype=tf.float32)
    #input_state = tf.Variable(cell1.zero_state(1,dtype=tf.float32),trainable=False)

    y_conv,output_state1,output_state2,output_state3,output_state4,diff = CNN(cell1, input_state1,cell2, input_state2,cell3, input_state3,cell4, input_state4, image1, keep_prob)
    loss2 = tf.sqrt(tf.reduce_mean(tf.square(y_conv - y_r)))
    #train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)
    loss = tf.reduce_mean(tf.image.ssim(y_r, y_conv, 1.0))
    loss_gen = - tf.reduce_sum(y_r * tf.log( tf.clip_by_value(y_conv,1e-20,1e+20)) + (1.-y_r) * tf.log( tf.clip_by_value(1.-y_conv,1e-20,1e+20)))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss_gen)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1.0)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,"model/m3_3_1.ckpt")


    filenames = ["ouput_2019-07-06-03-05-06"]

    train_image_list = make_datalist(filenames)

    filenames = ["ouput_2019-07-06-03-05-06"]
    test_image_list = make_datalist(filenames)
    print ("finish making dataset list")

    test_generator = GeneratorBuilder(test_image_list, batchSize)

    test_new_generator = BackgroundGenerator(
        test_generator.build(), max_prefetch=100)

    while(l <= 0):
        if((l + 1) % 1 == 0):
            trainrmse = 0
            rmse = 0
            num = 0
            result = []
            print("start test data test")
            for i, test_data in enumerate(test_new_generator):
                if not test_data:
                    break
                else:
                    test_image = test_data[0]
                    test_r = test_data[1]

                    loss_, states_1, states_2, states_3, states_4,output_image = sess.run([loss,output_state1,output_state2,output_state3,output_state4,y_conv], feed_dict={
                                 image1: [test_image[:,0,:,:,:]], y_r: [test_r[:,1,:,:,:]], keep_prob: 1.0})
                    result.append(output_image)
                    rmse = rmse + loss_
                    state_test1 = states_1
                    state_test2 = states_2
                    state_test3 = states_3
                    state_test4 = states_4

                    for m in range(8):
                        if(m <= 3):
                           loss_, states_1, states_2, states_3, states_4,output_image = sess.run([loss,output_state1,output_state2,output_state3,output_state4,y_conv], feed_dict={
                                 image1: [test_image[:,m+1,:,:,:]], input_state1:state_test1,input_state2:state_test2,input_state3:state_test3,input_state4:state_test4, y_r: [test_r[:,m+2,:,:,:]], keep_prob: 1.0})
                        else:
                           loss_, states_1, states_2, states_3, states_4,output_image = sess.run([loss,output_state1,output_state2,output_state3,output_state4,y_conv], feed_dict={
                                 image1: [(output_image > 0.1) * 1.0], input_state1:state_test1,input_state2:state_test2,input_state3:state_test3,input_state4:state_test4, y_r: [test_r[:,m+2,:,:,:]], keep_prob: 1.0})

                        result.append(output_image)
                        rmse = rmse + loss_
                        state_test1 = states_1
                        state_test2 = states_2
                        state_test3 = states_3
                        state_test4 = states_4
                        
                    num += 1
                    break
            rmse = rmse / (num*9)
            print (rmse)
            for i in range(9):
                cv2.imwrite("result/"+str(i)+".png",(result[i][0,:,:,:] > 0.5)*255)
            
        l += 1
