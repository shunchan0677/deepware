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


def CNN(image):
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

    h_fc4_drop = tf.nn.dropout(h_fc4, 1.0)

    with tf.name_scope('fc5') as scope:
        W_fc5 = weight_variable([10,2])
        b_fc5 = bias_variable([2])
        y_conv = tf.matmul(h_fc4_drop, W_fc5) + b_fc5


    return y_conv 

class RosTensorFlow():
    def __init__(self):
	self.sub_image0_flag=0
	self.sub_twist_flag=0
        self._cv_bridge = CvBridge()
        self._sub = rospy.Subscriber('/image_raw', Image, self.callback0, queue_size=1)
        self._pub = rospy.Publisher('/twist_raw', TwistStamped, queue_size=1)

        self.x = tf.placeholder(tf.float32, [None,66,200,3], name="x")

        self.y_conv  = CNN(self.x)

        self._session = tf.InteractiveSession()
        self._saver = tf.train.Saver()
        self._session.run(tf.global_variables_initializer())
        self._saver.restore(self._session, "carla_models2/bojarski_carla_tune.ckpt")

    def callback(self):

       if(True):

            self.sub_image0_flag=0
            self.sub_twist_flag=0
            list_image = []
            cv_image = self._cv_bridge.imgmsg_to_cv2(self.image_msg0, "bgr8")
            cv_image = cv2.resize(cv_image,(1920,1440))
            cv_image = cv_image[456:984,560:1360,:]
            cv_image = cv2.resize(cv_image,(200,66))/255.0
            list_image.append(cv_image)
            """
	    v = self.twist_msg0.twist.linear.x
	    w = self.twist_msg0.twist.angular.z/v
            x = 1/w - np.sqrt(1/(w*w) - v*v)
            y = v
	    l = np.sqrt(x*x + y*y)
	    label = [[x/l,y/l]]
            """

            result = self._session.run(self.y_conv, feed_dict={self.x:list_image})
            print result
            x = result[0][0]
            y = result[0][1]
            l = x*x + y*y
            if(x < 0.1):
                self._stre = 0.0
            else:
                self._stre = -y/l*x*4.0
            self._ac = x*4.0
            twist = TwistStamped()
            stre = self._stre
            ac = self._ac
            twist.twist.linear.x = ac
            twist.twist.angular.z = stre
            self._pub.publish(twist)

	    print stre


    def callback0(self,image_msg):
	    self.sub_image0_flag=1
	    self.image_msg0 = image_msg
	    self.callback()

    def callback1(self,twist_msg):
	    self.sub_twist_flag=1
	    self.twist_msg0 = twist_msg
	    self.callback()

    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('rostensorflow')
    tensor = RosTensorFlow()
    tensor.main()
