#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import tf as ros_tf
import tensorflow as tf
import numpy as np
import cv2
import random
import glob
import threading
import sys


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

    def build(self):
        while True:
            self.random_seq = range(len(self.image_label))
            random.shuffle(self.random_seq)
            for l in range(len(self.image_label)):
                image = cv2.imread(self.image_label[self.random_seq[l]])
                self.train_image.append(image)
                self.train_r.append(self.r_label[self.random_seq[l]])
                if((l + 1) % self.batch_size == 0):
                    self.train_image = np.asarray(self.train_image) / 255.0
                    self.train_r = np.asarray(
                        self.train_r).reshape(len(self.train_r), 20)
                    yield [self.train_image, self.train_r]
                    self.train_image = []
                    self.train_r = []
            yield None


def make_datalist(filenames):
    train_image_label = []
    train_r_label = []

    for filename in filenames:
        image_file = glob.glob("/home/brainiv/CIA/" +
                               filename + "/images/*.jpg")
        image_file.sort()
        r = np.loadtxt("/home/brainiv/CIA/" + filename +
                       "/tf.csv", delimiter=",")
        for l in range(len(image_file) - 11):
            train_image_label.append(image_file[l])
            q = r[l, 3:7]
            yaw = ros_tf.transformations.euler_from_quaternion(
                (q[0], q[1], q[2], q[3]))[2]
            rot_mat = np.array(
                [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            rot_res = np.dot(r[l + 1:l + 11, 0:2] - r[l, 0:2], rot_mat)
            train_r_label.append(rot_res)
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


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def CNN(image, keep_prob):
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 24])
        b_conv1 = bias_variable([24])
        h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
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

    with tf.name_scope('conv6') as scope:
        W_conv6 = weight_variable([3, 3, 64, 128])
        b_conv6 = bias_variable([128])
        h_conv6 = tf.nn.relu(conv2d(h_pool5, W_conv6) + b_conv6)
        h_pool6 = max_pool_2x2(h_conv6)

    with tf.name_scope('conv7') as scope:
        W_conv7 = weight_variable([3, 3, 128, 256])
        b_conv7 = bias_variable([256])
        h_conv7 = tf.nn.relu(conv2d(h_pool6, W_conv7) + b_conv7)
        h_pool7 = max_pool_2x2(h_conv7)

    h_conv7_flatten = tf.reshape(h_conv7, [-1, 7 * 7 * 256])

    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([7 * 7 * 256, 1164])
        b_fc1 = bias_variable([1164])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv7_flatten, W_fc1) + b_fc1)

    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1164, 100])
        b_fc2 = bias_variable([100])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    with tf.name_scope('fc3') as scope:
        W_fc3 = weight_variable([100, 20])
        b_fc3 = bias_variable([20])
        y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

    return y_conv


if __name__ == '__main__':
    l = 0
    basermse = 30000
    batchSize = 1

    image1 = tf.placeholder(tf.float32, [None, 400, 400, 3])
    y_r = tf.placeholder(tf.float32, [None, 20])
    keep_prob = tf.placeholder(tf.float32)

    y_conv = CNN(image1, keep_prob)
    loss = tf.sqrt(tf.reduce_mean(tf.square(y_conv - y_r)))
    train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1.0)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    filenames = ["output"]

    train_image_list, train_r_list = make_datalist(filenames)

    filenames = ["output"]
    test_image_list, test_r_list = make_datalist(filenames)
    print ("finish making dataset list")

    train_generator = GeneratorBuilder(
        train_image_list, train_r_list, batchSize)
    test_generator = GeneratorBuilder(test_image_list, test_r_list, batchSize)

    train_new_generator = BackgroundGenerator(
        train_generator.build(), max_prefetch=100)
    test_new_generator = BackgroundGenerator(
        test_generator.build(), max_prefetch=100)

    while(l <= 2000):
        print("start training")
        for i, train_data in enumerate(train_new_generator):
            if not train_data:
                break
            else:
                # print("Next!")
                train_image = train_data[0]
                train_r = train_data[1]
                sess.run(train_step, feed_dict={
                         image1: train_image, y_r: train_r, keep_prob: 0.5})
                if(i % 1000 == 0):
                    print("train_step:" + str(i))
        if((l + 1) % 1 == 0):
            trainrmse = 0
            rmse = 0
            """
            print("start training data test")
            for i,train_data in enumerate(train_new_generator):
                if not train_data:
                    break
                else:
                    train_image = train_data[0]
                    train_r = train_data[1]
                    trainrmse = trainrmse + sess.run(loss, feed_dict={image1: train_image, y_r: train_r,keep_prob:1.0})
            trainrmse = trainrmse/i
            """
            print("start test data test")
            for i, test_data in enumerate(test_new_generator):
                if not test_data:
                    break
                else:
                    test_image = test_data[0]
                    test_r = test_data[1]
                    rmse = rmse + \
                        sess.run(loss, feed_dict={
                                 image1: test_image, y_r: test_r, keep_prob: 1.0})
            rmse = rmse / i

            f = open("loss/model_m0.txt", "a")

            if(rmse < basermse):
                basermse = rmse
                save_path = saver.save(sess, "model/m0.ckpt")
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
