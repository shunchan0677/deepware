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
    def __init__(self, image_label, r_label, batch_size):
        self.train_image = []
        self.train_r = []
        self.image_label = image_label
        self.r_label = r_label
        self.batch_size = batch_size
        self.num_flame = 10

    def build(self):
        while True:
            self.random_seq = range(len(self.image_label))
            random.shuffle(self.random_seq)
            for l in range(len(self.image_label)):
                image = np.zeros((13,256,256,1))
                for flame in range(self.num_flame):
                    image[flame,:,:,:] = np.asarray(cv2.imread(self.image_label[self.random_seq[l]]+"/occupancy_grid_"+str(flame)+"_.jpg",cv2.IMREAD_GRAYSCALE)).reshape((1,256,256,1))/ 255.0
                image[10,:,:,:] = np.asarray(cv2.imread(self.image_label[self.random_seq[l]]+"/_vector_image_raw_ego_vehicle_.jpg",cv2.IMREAD_GRAYSCALE)).reshape((1,256,256,1))/ 255.0
                image[11,:,:,:] = np.asarray(cv2.imread(self.image_label[self.random_seq[l]]+"/_vector_image_raw_hd_map_.jpg",cv2.IMREAD_GRAYSCALE)).reshape((1,256,256,1))/ 255.0
                image[12,:,:,:] = np.asarray(cv2.imread(self.image_label[self.random_seq[l]]+"/_vector_image_raw_waypoint_.jpg",cv2.IMREAD_GRAYSCALE)).reshape((1,256,256,1))/ 255.0
                self.train_image.append(image)
                self.train_r.append(self.r_label[self.random_seq[l],3:12])
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
    print inputs.shape
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

    h_conv5_flatten = tf.reshape(h_pool5, [-1, 8 * 8 * 64])

    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([8 * 8 * 64, 1164])
        b_fc1 = bias_variable([1164])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flatten, W_fc1) + b_fc1)

    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1164, 100])
        b_fc2 = bias_variable([100])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    with tf.name_scope('fc3') as scope:
        W_fc3 = weight_variable([100, 9])
        b_fc3 = bias_variable([9])
        y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

    return y_conv


if __name__ == '__main__':
    l = 0
    basermse = 1000
    batchSize = 1

    OGM_x = tf.placeholder(tf.float32, [None,10] + [256, 256]+ [1])
    image1 = tf.placeholder(tf.float32, [None,3] + [256, 256]+ [1])
    OGM_y = tf.placeholder(tf.float32, [None,10] + [256, 256]+ [1])
    y_r = tf.placeholder(tf.float32, [None,9])
    keep_prob = tf.placeholder(tf.float32)

    OGM_pred = OGMPred(OGM_x)
    OGM_pred1 = tf.where(OGM_pred > 0.1,tf.ones([10,256,256,1]),tf.zeros([10,256,256,1]))
    y_conv = AgentNet(OGM_pred1, image1, keep_prob)

    reg_loss = tf.sqrt(tf.reduce_mean(tf.square(y_r - y_conv)))
    ssim_loss = tf.reduce_mean(tf.image.ssim(OGM_y, OGM_pred, 1.0))
    gen_loss = - tf.reduce_sum(OGM_y * tf.log( tf.clip_by_value(OGM_pred,1e-20,1e+20)) + (1.-OGM_y) * tf.log( tf.clip_by_value(1.-OGM_pred,1e-20,1e+20)))

    train_step = tf.train.AdamOptimizer(1e-3).minimize(tf.reduce_mean(gen_loss+reg_loss*100000))

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1.0)
    sess = tf.InteractiveSession()
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
        for i, train_data in enumerate(train_new_generator):
            if not train_data:
                break
            else:
                # print("Next!")
                train_image = train_data[0]
                train_r = train_data[1]
                train_image_result = copy.copy(train_image)
                train_image_result[:,5,:,:,:] = np.zeros((1,256,256,1))
                train_image_result[:,6,:,:,:] = np.zeros((1,256,256,1))
                train_image_result[:,7,:,:,:] = np.zeros((1,256,256,1))
                train_image_result[:,8,:,:,:] = np.zeros((1,256,256,1))
                train_image_result[:,9,:,:,:] = np.zeros((1,256,256,1))
                loss_tmp = sess.run([train_step,ssim_loss,reg_loss,gen_loss], feed_dict={
                         OGM_x:train_image_result[:,0:10,:,:,:],OGM_y:train_image[:,0:10,:,:,:],image1: train_image[:,10:13,:,:,:], y_r: train_r, keep_prob: 0.5})
                loss_a += loss_tmp[1]
                loss_b += loss_tmp[2]
                loss_c += loss_tmp[3]
                if(i % 1000 == 0 and i != 0):
                    print("train_step: " + str(i) +", ssim: "+str(loss_a/10)+", reg_loss: "+str(loss_b/1000)+", gen_loss: "+str(loss_c/1000))
                    loss_a = 0
                    loss_b = 0
                    loss_c = 0
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
                    test_image_result = test_image
                    test_image_result[:,5,:,:,:] = np.zeros((1,256,256,1))
                    test_image_result[:,6,:,:,:] = np.zeros((1,256,256,1))
                    test_image_result[:,7,:,:,:] = np.zeros((1,256,256,1))
                    test_image_result[:,8,:,:,:] = np.zeros((1,256,256,1))
                    test_image_result[:,9,:,:,:] = np.zeros((1,256,256,1))
                    rmse = rmse + \
                        sess.run(gen_loss, feed_dict={OGM_x:test_image_result[:,0:10,:,:,:],OGM_y:test_image[:,0:10,:,:,:],image1: test_image, y_r: test_r, keep_prob: 1.0})
                    num += 1
            rmse = rmse / num

            f = open("loss/model_m0_3.txt", "a")

            if(True):
                basermse = rmse
                save_path = saver.save(sess, "model/m0_3_"+str(l)+".ckpt")
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
