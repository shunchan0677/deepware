import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import Joy
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped


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


class RosTensorFlow():
    def __init__(self):
        self.sub_image0_flag = 0
        self._cv_bridge = CvBridge()
        self._sub = rospy.Subscriber(
            '/vector_image_raw', Image, self.callback_image)
        self.x = tf.placeholder(tf.float32, [None, 400, 400, 3], name="x")

        self.y_conv = CNN(self.x, 1.0)
        self._session = tf.InteractiveSession()
        self._saver = tf.train.Saver()
        self._session.run(tf.global_variables_initializer())
        self._saver.restore(self._session, "model/m0.ckpt")

    def callback(self):
        list_image = []
        cv_image = self._cv_bridge.imgmsg_to_cv2(self.image_msg0, "bgr8")
        list_image.append(cv_image)
        stre = self._session.run(self.y_conv, feed_dict={self.x: list_image})
        print(stre[0, :])

    def callback_image(self, image_msg):
        self.sub_image0_flag = 1
        self.image_msg0 = image_msg
        self.callback()

    def main(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('rostensorflow')
    tensor = RosTensorFlow()
    tensor.main()
