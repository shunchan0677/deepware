#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
import random
import glob
import threading
import sys
from cell2 import ConvLSTMCell
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
                    self.train_image[:,5,:,:,:] = np.zeros((1,256,256,1))
                    self.train_image[:,6,:,:,:] = np.zeros((1,256,256,1))
                    self.train_image[:,7,:,:,:] = np.zeros((1,256,256,1))
                    self.train_image[:,8,:,:,:] = np.zeros((1,256,256,1))
                    self.train_image[:,9,:,:,:] = np.zeros((1,256,256,1))
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
        train_image_label=image_file[1330:2100]
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


def CNN(image, keep_prob):
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
        print(h_conv3.shape)

    with tf.name_scope('dec-conv2') as scope:
        W_conv4 = weight_variable([3, 3, 1, 16]) 
        b_conv4 = bias_variable([1])
        h_conv4 = tf.sigmoid(deconv2d(h_conv3, W_conv4,256,1,10) + b_conv4) 
        print(h_conv4.shape)

    return h_conv4


if __name__ == '__main__':
    l = 0
    basermse = 30000
    batchSize = 1

    image1 = tf.placeholder(tf.float32, [None,10] + [256, 256]+ [1])
    y_r = tf.placeholder(tf.float32, [None,10] + [256, 256]+ [1])
    keep_prob = tf.placeholder(tf.float32)

    y_conv = CNN(image1, keep_prob)
    #loss = tf.sqrt(tf.reduce_mean(tf.square(y_conv - y_r)))
    #train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)
    loss = tf.reduce_mean(tf.image.ssim(y_r, y_conv, 2.0))
    train_step = tf.train.AdamOptimizer(1e-2).minimize(-1*loss)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1.0)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,"model/m0_3_4.ckpt")

    filenames = ["2019-07-13-13-35-50"]

    train_image_list = make_datalist(filenames)

    print ("finish making dataset list")

    train_generator = GeneratorBuilder(
        train_image_list, batchSize)

    train_new_generator = BackgroundGenerator(
        train_generator.build(), max_prefetch=100)

    while(l <= 0):
        print("start training")
        for i, train_data in enumerate(train_new_generator):
            if not train_data:
                break
            else:
                print("Next!")
                train_image = train_data[0]
                train_r = train_data[1]
                result = sess.run(y_conv, feed_dict={
                         image1: train_image, y_r: train_r, keep_prob: 0.5})
                loss = sess.run(loss, feed_dict={
                         image1: train_image, y_r: train_r, keep_prob: 0.5})
                print(result.shape,train_image.shape,loss)
                break
        l += 1
    for i in range(10):
       cv2.imwrite("result/"+str(i)+".png",(result[i,:,:,:] > 0.1)*255)
