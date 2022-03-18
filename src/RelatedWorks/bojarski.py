#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import tensorflow as tf
import numpy as np
import cv2
import random
import glob
import numpy as np
import threading

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
                path = self.image_label[self.random_seq[l]]+"/image.png"
                if(os.path.exists(path) == False):
                    flag = False

                if(flag):
                    image = cv2.imread(self.image_label[self.random_seq[l]]+"/image.png")
                    self.train_image.append(image)
                    self.train_r.append(self.r_label[self.random_seq[l]][12:14])
                    if((l + 1) % self.batch_size == 0):
                        self.train_image = np.asarray(self.train_image)/ 255.0
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
    # --- define weight
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # --- define bias
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv1d(x, W):
    # --- define convolution
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')

def conv2d(x, W):
    # --- define convolution
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def lrelu(x):
    x1 = tf.maximum(x,x/5.5)
    return x1


def CNN(image,keep_prob):
    with tf.name_scope('norm1') as scope:
        norm1=tf.nn.lrn(image,4,bias=1.0,alpha=0.001/9.0,beta=0.75)

    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 24])
        b_conv1 = bias_variable([24])
        h_conv1 = tf.nn.relu(conv1d(norm1, W_conv1) + b_conv1)

    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 24, 36])
        b_conv2 = bias_variable([36])
        h_conv2 = tf.nn.relu(conv1d(h_conv1, W_conv2) + b_conv2)

    with tf.name_scope('conv3') as scope:
        W_conv3 = weight_variable([5, 5, 36, 48])
        b_conv3 = bias_variable([48])
        h_conv3 = tf.nn.relu(conv1d(h_conv2, W_conv3) + b_conv3)

    with tf.name_scope('conv4') as scope:
        W_conv4 = weight_variable([3, 3, 48, 64])
        b_conv4 = bias_variable([64])
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

    with tf.name_scope('conv5') as scope:
        W_conv5 = weight_variable([3, 3, 64, 64])
        b_conv5 = bias_variable([64])
        h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)

    h_conv5_flatten = tf.reshape(h_conv5, [-1,1152])

    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([1152, 1164])
        b_fc1 = bias_variable([1164])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flatten, W_fc1) + b_fc1)

    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1164,100])
        b_fc2 = bias_variable([100])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    with tf.name_scope('fc3') as scope:
        W_fc3 = weight_variable([100,50])
        b_fc3 = bias_variable([50])
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

    with tf.name_scope('fc4') as scope:
        W_fc4 = weight_variable([50,10])
        b_fc4 = bias_variable([10])
        h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)

    h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

    with tf.name_scope('fc5') as scope:
        W_fc5 = weight_variable([10,2])
        b_fc5 = bias_variable([2])
        y_conv = tf.matmul(h_fc4_drop, W_fc5) + b_fc5


    return y_conv 

if __name__ == '__main__':
    l=0
    basermse = 5
    batchSize = 1

    image1 = tf.placeholder(tf.float32, [None, 66, 200, 3])
    y_r = tf.placeholder(tf.float32, [None, 2])
    keep_prob = tf.placeholder(tf.float32)

    y_conv  = CNN(image1,keep_prob)
    loss = tf.sqrt(tf.reduce_sum(tf.square(y_conv-y_r)))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(loss)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours = 1.0)
    sess = tf.InteractiveSession()
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

    train_generator = GeneratorBuilder(train_image_list, train_r_label, batchSize)
    valid_generator = GeneratorBuilder(valid_image_list, valid_r_label, batchSize)
    test_generator = GeneratorBuilder(test_image_list, test_r_label, batchSize)

    train_new_generator = BackgroundGenerator(
        train_generator.build(), max_prefetch=100)
    valid_new_generator = BackgroundGenerator(
        valid_generator.build(), max_prefetch=100)
    test_new_generator = BackgroundGenerator(
        test_generator.build(), max_prefetch=100)

    print ("finish making dataset list")


    while(l<=2000):
        print("start training")
        for i,train_data in enumerate(train_new_generator):
            if not train_data:
                break
            else:
                #print("Next!")
                train_image = train_data[0]
                train_r  = train_data[1]
                sess.run(train_step,feed_dict={image1:train_image, y_r:train_r, keep_prob: 0.5})
                #print("train_step:" + str(i)) 
        if((l+1)%1 == 0):
            valid_rmse = 0
            rmse = 0

            print("start valid data test")
            for i,valid_data in enumerate(valid_new_generator):
                if not valid_data:
                    break
                else:
                    test_image = valid_data[0]
                    test_r = valid_data[1]
                    valid_rmse = valid_rmse + sess.run(loss, feed_dict={image1: test_image, y_r: test_r,keep_prob:1.0})
            valid_rmse = valid_rmse/i

            print("start test data test")
            for i,test_data in enumerate(test_new_generator):
                if not test_data:
                    break
                else:
                    test_image = test_data[0]
                    test_r = test_data[1]
                    rmse = rmse + sess.run(loss, feed_dict={image1: test_image, y_r: test_r,keep_prob:1.0})
            rmse = rmse/i

            f = open("loss/bojarski.txt","a")

            if(valid_rmse<basermse):
                basermse=valid_rmse
                save_path = saver.save(sess, "model/bojarski.ckpt")
                f.write("epoch:=%d , train_rmse:=%f, valid_rmse:=%f, test_rmse:=%f ModelSave!\n" % (
                    l, basermse, valid_rmse, rmse))
                print("epoch:=%d , train_rmse:=%f, valid_rmse:=%f ,test_rmse:=%f ModelSave!\n" %
                      (l, basermse, valid_rmse, rmse))
            else:
                f.write("epoch:=%d ,train_rmse:=%f, valid_rmse:=%f, test_rmse:=%f\n" %
                        (l, basermse, valid_rmse, rmse))
                print("epoch:=%d ,train_rmse:=%f, valid_rmse:=%f, test_rmse:=%f\n" %
                      (l, basermse, valid_rmse, rmse))
            f.close()
        l += 1

