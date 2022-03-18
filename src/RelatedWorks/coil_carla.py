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
        self.train_flga = []
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
                path = self.image_label[self.random_seq[l]]+"/image_raw_.jpg"
                if(os.path.exists(path) == False):
                    flag = False

                if(flag):
                    image = cv2.imread(self.image_label[self.random_seq[l]]+"/image_raw_.jpg")
                    self.train_image.append(image)
                    x = self.r_label[self.random_seq[l]][27]
                    y = self.r_label[self.random_seq[l]][28]
                    yaw = self.r_label[self.random_seq[l]][29]
                    yaw_4 = self.r_label[self.random_seq[l]][14]
                    diff_x = x*np.cos(-yaw) - y*np.sin(-yaw)
                    diff_y = x*np.sin(-yaw) + y*np.cos(-yaw)
                    x = diff_x*np.cos(yaw_4) - diff_y*np.sin(yaw_4)
                    y = diff_x*np.sin(yaw_4) + diff_y*np.cos(yaw_4)
                    if(x<0):
                        x = -x
                        y = -y
                    #print(x,y)
                    self.train_r.append([x,y])

                    self.train_r.append(self.r_label[self.random_seq[l]][27:29])
                    way_image = cv2.imread(self.image_label[self.random_seq[l]]+"/way_image_.jpg",cv2.IMREAD_GRAYSCALE)/255.0
                    if(np.any(way_image[10:100,:200]>0.1)):
                        flags =  np.asarray([1,0,0])
                    elif(np.any(way_image[300:390,:200]>0.1)):
                        flags =  np.asarray([0,0,1])
                    else:
                        flags =  np.asarray([0,1,0])
                    if((l + 1) % self.batch_size == 0):
                        self.train_image = np.asarray(self.train_image)/ 255.0
                        if(x<0.2):
                            image_a,x_r,y_r = image_rotation(self.image_label[self.random_seq[l]]+"/image_raw_.jpg", x,y,4)
                        else:
                            i = random.randint(0,8)
                            image_a,x_r,y_r = image_rotation(self.image_label[self.random_seq[l]]+"/image_raw_.jpg", x,y,i)
                        self.train_image = np.asarray([image_a])
                        self.train_r = []
                        self.train_r.append([x_r,y_r])
                        yield [self.train_image, self.train_r,flags]
                        self.train_image = []
                        self.train_r = []
            yield None


cameraMatrix1 = np.loadtxt('../cameraMatrix0.csv',delimiter = ',')
cameraMatrix2 = np.loadtxt('../cameraMatrix0.csv',delimiter = ',')
distCoeffs2 = np.loadtxt('../distCoeffs0.csv',delimiter = ',')


def make_datalist(filenames):
    train_image_label = []
    train_r_label = []

    for filename in filenames:
        image_file = glob.glob("/home/s_seiya/tmp/CARLA_dataset/" +
                               filename + "/images/*")
        image_file.sort()

        r = np.loadtxt("/home/s_seiya/tmp/CARLA_dataset/" + filename +
                       "/tf.csv", delimiter=",").tolist()

        train_image_label+=image_file[:len(r)]
        train_r_label += r

    print(len(train_image_label),len(train_r_label))
    return train_image_label, train_r_label

def image_rotation(path,x,y,i):
    image = cv2.imread(path)
    #i = random.randint(0,8)
    if(path.split("/")[-1]=="image_raw_.jpg"):
        image = cv2.resize(image,(1920,1440))
        newimageSize = (image.shape[1],image.shape[0])
        r=np.pi/180*5*(4-i)
        x_r = float(x)*np.cos(-r) - float(y)*np.sin(-r)
        y_r = float(x)*np.sin(-r) + float(y)*np.cos(-r)
        R = np.matrix(((np.cos(r),0.,-np.sin(r)),(0.,1.,0.),(np.sin(r),0.,np.cos(r))))
        map1_l, map2_l = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R,cameraMatrix1, newimageSize, cv2.CV_32FC1)
        interpolation = cv2.INTER_LINEAR
        image = cv2.remap(image, map1_l, map2_l, interpolation) 
        image = image[456:984,560:1360,:]
        image = cv2.resize(image,(200,66))
    else:
        center = (int(400/2), int(320))
        angle = 5*(4-i)
        scale = 1.0
        vehicle_x = 7/2
        vehicle_y = 3.8/2
        if(path.split("/")[-1]=="occupancy_grid_4_.jpg" or path.split("/")[-1]=="obj_bbox_4_.jpg" or path.split("/")[-1]=="ego_bbox_4_.jpg"):
            egovehicle = copy.copy(image[320-vehicle_x*400/80:320+vehicle_x*400/80 ,200-vehicle_y*400/80:200+vehicle_y*400/80])
            image[320-vehicle_x*400/80:320+vehicle_x*400/80 ,200-vehicle_y*400/80:200+vehicle_y*400/80] = 0
        trans = cv2.getRotationMatrix2D(center, angle , scale)
        image = cv2.warpAffine(image, trans, (400,400))
        if(path.split("/")[-1]=="occupancy_grid_4_.jpg" or path.split("/")[-1]=="obj_bbox_4_.jpg" or path.split("/")[-1]=="ego_bbox_4_.jpg"):
            image[320-vehicle_x*400/80:320+vehicle_x*400/80 ,200-vehicle_y*400/80:200+vehicle_y*400/80] = egovehicle
    return image/255.0,x_r,y_r

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


def CNN(image,flag, keep_prob):
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

    with tf.name_scope('fc5_r') as scope:
        W_fc5 = weight_variable([10,2])
        b_fc5 = bias_variable([2])
        y_conv_r = tf.matmul(h_fc4_drop, W_fc5) + b_fc5


    with tf.name_scope('fc5_s') as scope:
        W_fc5_s = weight_variable([10,2])
        b_fc5_s = bias_variable([2])
        y_conv_s = tf.matmul(h_fc4_drop, W_fc5_s) + b_fc5_s


    with tf.name_scope('fc5_l') as scope:
        W_fc5_l = weight_variable([10,2])
        b_fc5_l = bias_variable([2])
        y_conv_l = tf.matmul(h_fc4_drop, W_fc5_l) + b_fc5_l


    y_conv = y_conv_s
    y_conv = tf.where(flag[0] == 1, y_conv, y_conv_l)
    y_conv = tf.where(flag[2] == 1, y_conv, y_conv_r)

    return y_conv 

if __name__ == '__main__':
    l=0
    basermse = 5
    batchSize = 1

    image1 = tf.placeholder(tf.float32, [None, 66, 200, 3])
    y_r = tf.placeholder(tf.float32, [None, 2])
    flag = tf.placeholder(tf.int32, [3])
    keep_prob = tf.placeholder(tf.float32)

    y_conv  = CNN(image1,flag, keep_prob)
    loss = tf.sqrt(tf.reduce_sum(tf.square(y_conv-y_r)))
    train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours = 1.0)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())


    filenames = ["_2019-09-04-05-57-49","_2019-09-04-06-08-39","_2019-09-04-06-16-03","_2019-09-04-06-29-08","_2019-09-04-06-46-23","_2019-09-04-06-56-23"]

    train_image_list,train_r_label = make_datalist(filenames)

    filenames = ["_2019-09-04-07-24-55"]
    valid_image_list,valid_r_label = make_datalist(filenames)

    filenames = ["_2019-09-04-07-24-55"]
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
                train_flag = train_data[2]
                sess.run(train_step,feed_dict={image1:train_image, flag:train_flag,  y_r:train_r, keep_prob: 0.5})
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
                    test_flag = valid_data[2]
                    valid_rmse = valid_rmse + sess.run(loss, feed_dict={image1: test_image, flag: test_flag ,y_r: test_r,keep_prob:1.0})
            valid_rmse = valid_rmse/i

            print("start test data test")
            for i,test_data in enumerate(test_new_generator):
                if not test_data:
                    break
                else:
                    test_image = test_data[0]
                    test_r = test_data[1]
                    test_flag = test_data[2]
                    rmse = rmse + sess.run(loss, feed_dict={image1: test_image, flag: test_flag, y_r: test_r,keep_prob:1.0})
            rmse = rmse/i

            f = open("loss/coil_carla.txt","a")

            if(valid_rmse<basermse):
                basermse=valid_rmse
                save_path = saver.save(sess, "model/coil_carla.ckpt")
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

