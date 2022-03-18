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
from ogm import im2ogm

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
            random.shuffle(self.random_seq)
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
        train_image_label=image_file
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


def CNN(cell1, input_state1, cell2, input_state2, cell3, input_state3, cell4, input_state4, cell5, input_state5, image, keep_prob):
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

    with tf.name_scope('core-RNN-conv2') as scope:
        outputs2, state2 = tf.nn.dynamic_rnn(cell=cell2, inputs=outputs1, dtype=outputs1.dtype, initial_state=input_state2)

    with tf.name_scope('core-RNN-conv3') as scope:
        outputs3, state3 = tf.nn.dynamic_rnn(cell=cell3, inputs=outputs2, dtype=outputs2.dtype, initial_state=input_state3)

    with tf.name_scope('core-RNN-conv4') as scope:
        outputs4, state4 = tf.nn.dynamic_rnn(cell=cell4, inputs=outputs3, dtype=outputs3.dtype, initial_state=input_state4)


    outputs_ = tf.reshape(outputs4,[-1]+[64,64]+[32])

    with tf.name_scope('dec-conv1') as scope:
        W_conv3 = weight_variable([3, 3, 16, 32])
        b_conv3 = bias_variable([16])
        h_conv3 = tf.nn.relu(deconv2d(outputs_, W_conv3,128,16,1) + b_conv3)

    with tf.name_scope('dec-conv2') as scope:
        W_conv4 = weight_variable([3, 3, 1, 16]) 
        b_conv4 = bias_variable([1])
        h_conv4 = deconv2d(h_conv3, W_conv4,256,1,1) + b_conv4

    ok = tf.sigmoid(h_conv4+image)


    ok1 = tf.reshape(ok,[-1,1]+[256,256]+[1])

    with tf.name_scope('class-RNN-conv4') as scope:
        output5, state5 = tf.nn.dynamic_rnn(cell=cell5, inputs=ok1, dtype=ok1.dtype, initial_state=input_state5)

    output5_ = tf.reshape(output5,[-1]+[256,256]+[2])

    return output5_, state1,state2,state3,state4,state5, h_conv4

def linear(x):
    return x

def biased_softmax(logits,labels):
    labels_f = tf.cast(labels,tf.float32)
    logit_soft = tf.nn.softmax(logits)
    labels_2 = tf.one_hot(labels,2)
    cross_entropy = - tf.reduce_sum(tf.log(logit_soft)*tf.cast(labels_2,tf.float32),axis=1)
    white_rate = tf.reduce_sum(labels_f) / tf.cast(tf.shape(labels_f), tf.float32)
    biased_cross_entropy = (1-white_rate) * cross_entropy* labels_f + (white_rate)*cross_entropy* (1 - labels_f)
    print (biased_cross_entropy)

    return tf.cast(2.0*biased_cross_entropy,tf.float32)


def masked_biased_softmax(logits,labels,masks):
    masks = tf.cast(tf.reshape(masks,[-1]),tf.float32)
    minus_masks = 1.0 - tf.cast(tf.reshape(masks,[-1]),tf.float32)

    labels_f = tf.cast(labels,tf.float32)
    logit_soft = tf.nn.softmax(logits)
    labels_2 = tf.one_hot(labels,2)
    cross_entropy = tf.reduce_sum(tf.log(logit_soft)*tf.cast(labels_2,tf.float32),axis=1)
    masked_cross_entropy = minus_masks * cross_entropy
    white_rate1 = tf.reduce_sum(labels_f) - tf.reduce_sum(masks)
    white_rate2 = tf.cast(tf.shape(labels_f), tf.float32) - tf.reduce_sum(masks)
    white_rate = white_rate1/white_rate2
    biased_cross_entropy = (1-white_rate) * cross_entropy* labels_f + (white_rate)*cross_entropy* (1 - labels_f)

    return - tf.cast(2.0*biased_cross_entropy,tf.float32)

if __name__ == '__main__':
    l = 0
    basermse = 1000
    batchSize = 1

    image1 = tf.placeholder(tf.float32, [None,1] + [256, 256]+ [1])
    y_r = tf.placeholder(tf.float32, [None,1] + [256, 256]+ [1])
    y_mask = tf.placeholder(tf.float32, [None,1] + [256, 256]+ [1])
    keep_prob = tf.placeholder(tf.float32)
    
    cell1 = ConvLSTMCell([64,64], 32, [3, 3],"1")
    input_state1 = cell1.zero_state(1,dtype=tf.float32)

    cell2 = ConvLSTMCell([64,64], 32, [3, 3],"2")
    input_state2 = cell2.zero_state(1,dtype=tf.float32)

    cell3 = ConvLSTMCell([64,64], 32, [3, 3],"3")
    input_state3 = cell3.zero_state(1,dtype=tf.float32)

    cell4 = ConvLSTMCell([64,64], 32, [3, 3],"4")
    input_state4 = cell4.zero_state(1,dtype=tf.float32)

    cell5 = ConvLSTMCell([256,256], 2, [1, 1],"5",activation=linear)
    input_state5 = cell5.zero_state(1,dtype=tf.float32)
    #input_state = tf.Variable(cell1.zero_state(1,dtype=tf.float32),trainable=False)

    y_conv1,output_state1,output_state2,output_state3,output_state4,output_state5,diff = CNN(cell1, input_state1,cell2, input_state2,cell3, input_state3,cell4, input_state4, cell5, input_state5, image1, keep_prob)
    y_conv_soft = tf.nn.softmax(y_conv1)
    y_conv = tf.cast(tf.reshape(tf.argmax(y_conv_soft,axis=3),[-1,1,256,256,1]),tf.float32)
    y_conv1 = tf.cast(tf.reshape(y_conv_soft[:,:,:,1],[-1,1,256,256,1]),tf.float32)

    loss2 = tf.sqrt(tf.reduce_mean(tf.square(y_conv - y_r)))
    loss_ssim = tf.image.ssim(y_r, y_conv, 1.0)
    loss = tf.reduce_mean(loss_ssim)
    loss2 = tf.reduce_sum(loss_ssim)
    #train_step = tf.train.AdamOptimizer(1e-2).minimize(-1*loss)
    loss_gen1 = y_r * tf.log( tf.clip_by_value(y_conv1,1e-20,1e+20)) + (1.-y_r) * tf.log( tf.clip_by_value(1.-y_conv1,1e-20,1e+20))
    print(loss_gen1.shape)
    loss_gen = - tf.reduce_mean(loss_gen1)

    loss_pix = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reshape(y_conv1,[-1,2]), labels=tf.cast(tf.reshape(y_r,[-1]),tf.int32))
    loss_pix_b = biased_softmax(logits=tf.reshape(y_conv1,[-1,2]), labels=tf.cast(tf.reshape(y_r,[-1]),tf.int32))
    loss_pix_mask = masked_biased_softmax(logits=tf.reshape(y_conv1,[-1,2]), labels=tf.cast(tf.reshape(y_r,[-1]),tf.int32),masks = y_mask)
    loss_norm = tf.nn.l2_loss(diff)

    print(loss_pix_mask.shape)
    print(loss_ssim.shape)
    print(loss_norm.shape)
    print(loss_gen.shape)


    total_loss = loss_gen #tf.reduce_sum(loss_pix_mask) - tf.reduce_sum(loss) + tf.reduce_sum(loss_norm)
    print(total_loss.shape)

    train_step = tf.train.AdamOptimizer(1e-3).minimize(total_loss)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1.0)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    filenames = ["ouput_2019-07-06-03-05-06"]

    train_image_list = make_datalist(filenames)

    filenames = ["ouput_2019-07-06-02-10-11"]
    test_image_list = make_datalist(filenames)
    print ("finish making dataset list")

    train_generator = GeneratorBuilder(
        train_image_list, batchSize)
    test_generator = GeneratorBuilder(test_image_list, batchSize)

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
                #mid_state = tf.contrib.rnn.LSTMStateTuple(mid_state[0], mid_state[1])
                #mid_state = np.zeros((1,64,64,32))
                loss_tmp = sess.run([train_step,loss,loss2,loss_gen,output_state1,output_state2,output_state3,output_state4,output_state5,y_conv], feed_dict={
                              image1: [train_image[:,0,:,:,:]], y_r: [train_r[:,1,:,:,:]],y_mask: [im2ogm(train_r[:,1,:,:,:])], keep_prob: 0.5})
                loss_a += loss_tmp[1]
                loss_b += loss_tmp[2]
                loss_c += loss_tmp[3]
                mid_state1 = loss_tmp[4]
                mid_state2 = loss_tmp[5]
                mid_state3 = loss_tmp[6]
                mid_state4 = loss_tmp[7]
                mid_state5 = loss_tmp[8]
                output_image = (loss_tmp[9] > 0.5) * 1.0
                for m in range(8):
                   if(m <= 3):
                       loss_tmp = sess.run([train_step,loss,loss2,loss_gen,output_state1,output_state2,output_state3,output_state4,output_state5,y_conv], feed_dict={
                              image1: [train_image[:,m+1,:,:,:]],input_state1:mid_state1,input_state2:mid_state2,input_state3:mid_state3,input_state4:mid_state4,input_state5:mid_state5, y_r: [train_r[:,m+2,:,:,:]],y_mask: [im2ogm(train_r[:,m+2,:,:,:])], keep_prob: 0.5})
                       loss_a += loss_tmp[1]
                       loss_b += loss_tmp[2]
                       loss_c += loss_tmp[3]
                       mid_state1 = loss_tmp[4]
                       mid_state2 = loss_tmp[5]
                       mid_state3 = loss_tmp[6]
                       mid_state4 = loss_tmp[7]
                       mid_state5 = loss_tmp[8]
                       output_image = (loss_tmp[9] > 0.5) * 1.0
                   else:
                       loss_tmp = sess.run([train_step,loss,loss2,loss_gen,output_state1,output_state2,output_state3,output_state4,output_state5,y_conv], feed_dict={
                              image1: output_image,input_state1:mid_state1,input_state2:mid_state2,input_state3:mid_state3,input_state4:mid_state4,input_state5:mid_state5, y_r: [train_r[:,m+2,:,:,:]],y_mask: [im2ogm(train_r[:,m+2,:,:,:])], keep_prob: 0.5})
                       loss_a += loss_tmp[1]
                       loss_b += loss_tmp[2]
                       loss_c += loss_tmp[3]
                       mid_state1 = loss_tmp[4]
                       mid_state2 = loss_tmp[5]
                       mid_state3 = loss_tmp[6]
                       mid_state4 = loss_tmp[7]
                       mid_state5 = loss_tmp[8]
                       output_image = (loss_tmp[9] > 0.5) * 1.0
                if(i % 1000 == 0 and i != 0):
                    print("train_step: " + str(i) +", ssim: "+str(loss_a/90)+", loss: "+str(loss_b/9000)+", loss_gen: "+str(loss_c/9000))
                    loss_a = 0
                    loss_b = 0
                    loss_c = 0
        if((l + 1) % 1 == 0):
            trainrmse = 0
            rmse = 0
            num = 0
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

                    loss_, states_1, states_2, states_3, states_4, states_5 = sess.run([loss_gen,output_state1,output_state2,output_state3,output_state4,output_state5], feed_dict={
                                 image1: [test_image[:,0,:,:,:]], y_r: [test_r[:,1,:,:,:]],y_mask: [im2ogm(train_r[:,1,:,:,:])], keep_prob: 1.0})

                    rmse = rmse + loss_
                    state_test1 = states_1
                    state_test2 = states_2
                    state_test3 = states_3
                    state_test4 = states_4
                    state_test5 = states_5

                    for m in range(8):
                        if(m <= 3):
                           loss_, states_1, states_2, states_3, states_4,states_5,output_image = sess.run([loss,output_state1,output_state2,output_state3,output_state4,output_state5,y_conv], feed_dict={
                                 image1: [test_image[:,m+1,:,:,:]], input_state1:state_test1,input_state2:state_test2,input_state3:state_test3,input_state4:state_test4,input_state5:state_test5, y_r: [test_r[:,m+2,:,:,:]],y_mask: [im2ogm(train_r[:,m+2,:,:,:])], keep_prob: 1.0})
                        else:
                           loss_, states_1, states_2, states_3, states_4, states_5,output_image = sess.run([loss,output_state1,output_state2,output_state3,output_state4,output_state5,y_conv], feed_dict={
                                 image1: output_image, input_state1:state_test1,input_state2:state_test2,input_state3:state_test3,input_state4:state_test4,input_state5:state_test5, y_r: [test_r[:,m+2,:,:,:]],y_mask: [im2ogm(train_r[:,m+2,:,:,:])], keep_prob: 1.0})


                        rmse = rmse + loss_
                        state_test1 = states_1
                        state_test2 = states_2
                        state_test3 = states_3
                        state_test4 = states_4
                        state_test5 = states_5
                        
                    num += 1
            rmse = rmse / (num*9)

            f = open("loss/model_m6_3.txt", "a")

            if(True):
                basermse = rmse
                save_path = saver.save(sess, "model/m6_3_"+str(l)+".ckpt")
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
