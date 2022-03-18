import rospy
import tf as tf2
from sensor_msgs.msg import Image
from sensor_msgs.msg import Joy
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
from cell2 import ConvLSTMCell
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point


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
    #inputs = tf.where(inputs > 0.1,tf.ones([13,256,256,1]),tf.zeros([13,256,256,1]))
    #inputs = tf.reshape(inputs,[1,13]+[256,256]+[1])
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

    h_conv5_flatten = tf.reshape(h_pool5, [1, 13 * 8 * 8 * 64])

    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([13 * 8 * 8 * 64, 1164])
        b_fc1 = bias_variable([1164])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flatten, W_fc1) + b_fc1)

    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1164, 100])
        b_fc2 = bias_variable([100])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    with tf.name_scope('fc3') as scope:
        W_fc3 = weight_variable([100, 4])
        b_fc3 = bias_variable([4])
        y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

    return y_conv


class RosTensorFlow():
    def __init__(self):
        self._cv_bridge = CvBridge()
        self._sub_ego_vehicle = rospy.Subscriber('/vector_image_raw/ego_vehicle', Image, self.callback_ego_vehicle)
        self._sub_hd_map = rospy.Subscriber('/vector_image_raw/hd_map', Image, self.callback_hd_map)
        self._sub_waypoint = rospy.Subscriber('/vector_image_raw/waypoint', Image, self.callback_waypoint)
        self._sub_occupancy_grid_0 = rospy.Subscriber('/occupancy_grid_0', Image, self.callback_occupancy_grid_0)
        self._sub_occupancy_grid_1 = rospy.Subscriber('/occupancy_grid_1', Image, self.callback_occupancy_grid_1)
        self._sub_occupancy_grid_2 = rospy.Subscriber('/occupancy_grid_2', Image, self.callback_occupancy_grid_2)
        self._sub_occupancy_grid_3 = rospy.Subscriber('/occupancy_grid_3', Image, self.callback_occupancy_grid_3)
        self._sub_occupancy_grid_4 = rospy.Subscriber('/occupancy_grid_4', Image, self.callback_occupancy_grid_4)
        self.list_image = np.zeros((1,13,256,256,1))
        self.pub_res = rospy.Publisher('result_mark', MarkerArray, queue_size=1)
        self.listener = tf2.TransformListener()
        self.br = tf2.TransformBroadcaster()


        self.OGM_x = tf.placeholder(tf.float32, [None,10] + [256, 256]+ [1])
        self.image1 = tf.placeholder(tf.float32, [None,3] + [256, 256]+ [1])
        self.OGM_y = tf.placeholder(tf.float32, [None,10] + [256, 256]+ [1])
        self.y_r = tf.placeholder(tf.float32, [None,2])
        self.keep_prob = tf.placeholder(tf.float32)

        self.OGM_pred = OGMPred(self.OGM_x)
        self.OGM_pred1 = tf.where(self.OGM_pred > 0.1,tf.ones([10,256,256,1]),tf.zeros([10,256,256,1]))
        self.y_conv = AgentNet(self.OGM_pred1, self.image1, self.keep_prob)
        mean_x = self.y_conv[0][0]
        var_x = tf.math.abs(self.y_conv[0][1])
        mean_y = self.y_conv[0][2]
        var_y = tf.math.abs(self.y_conv[0][3])

        p_x = tfd.Normal(loc=mean_x,scale=var_x)
        p_y = tfd.Normal(loc=mean_y,scale=var_y)

        self.result_x = p_x.sample(10)
        self.result_y = p_y.sample(10)

        #loss_x = tf.math.abs(self.y_r[0][0] - self.result_x)
        #loss_y = tf.math.abs(self.y_r[0][1] - self.result_y)
 
        #likelihood_x = tf.reduce_sum(p_x.log_prob(self.y_r[0][0]))
        #likelihood_y = tf.reduce_sum(p_y.log_prob(self.y_r[0][1]))

        self.reg_loss = 0#loss_x + loss_y

        #self.reg_loss = tf.sqrt(tf.reduce_max(tf.square(self.y_r - self.y_conv)))
        self.ssim_loss = tf.reduce_mean(tf.image.ssim(self.OGM_y, self.OGM_pred, 1.0))
        self.gen_loss = - tf.reduce_sum(self.OGM_y * tf.log( tf.clip_by_value(self.OGM_pred,1e-20,1e+20)) + (1.-self.OGM_y) * tf.log( tf.clip_by_value(1.-self.OGM_pred,1e-20,1e+20)))

        self.train_step = tf.train.AdamOptimizer(1e-3).minimize(tf.reduce_mean(self.gen_loss+self.reg_loss*100000))

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1.0)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, "model/m0_10_1.ckpt")

    def callback_ego_vehicle(self,image):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image, "bgr8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (256,256))
        self.list_image[:,10,:,:,:] = gray.reshape((1,256,256,1))/ 255.0
        #print("ego_vehicle")


    def callback_hd_map(self,image):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image, "bgr8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (256,256))
        self.list_image[:,11,:,:,:] = gray.reshape((1,256,256,1))/ 255.0
        #print("hd_map")

    def callback_waypoint(self,image):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image, "bgr8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (256,256))
        self.list_image[:,12,:,:,:] = gray.reshape((1,256,256,1))/ 255.0
        #print("waypoint")

    def callback_occupancy_grid_0(self,image):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image, "bgr8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (256,256))
        self.list_image[:,0,:,:,:] = gray.reshape((1,256,256,1))/ 255.0
        #print("occupancy_grid_0")

    def callback_occupancy_grid_1(self,image):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image, "bgr8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (256,256))
        self.list_image[:,1,:,:,:] = gray.reshape((1,256,256,1))/ 255.0
        #print("occupancy_grid_1")

    def callback_occupancy_grid_2(self,image):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image, "bgr8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (256,256))
        self.list_image[:,2,:,:,:] = gray.reshape((1,256,256,1))/ 255.0
        #print("occupancy_grid_2")

    def callback_occupancy_grid_3(self,image):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image, "bgr8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (256,256))
        self.list_image[:,3,:,:,:] = gray.reshape((1,256,256,1))/ 255.0
        #print("occupancy_grid_3")

    def callback_occupancy_grid_4(self,image):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image, "bgr8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (256,256))
        self.list_image[:,4,:,:,:] = gray.reshape((1,256,256,1))/ 255.0
        #print("occupancy_grid_4")
        self.callback_model()

    def callback_model(self):
        pred_pose = self.sess.run([self.result_x,self.result_y], feed_dict={self.OGM_x:self.list_image[:,0:10,:,:,:]*1.0,self.image1: self.list_image[:,10:13,:,:,:]*1.0, self.keep_prob: 0.5})
        #print("model")
        #print(pred_pose)


        self.listener.waitForTransform("/world", "/base_link", rospy.Time.now(), rospy.Duration(4.0))
        (trans, rot) = self.listener.lookupTransformFull("/world", rospy.Time.now(), "/base_link", rospy.Time.now() - rospy.Duration((12-5)*0.1),"world")
        e = tf2.transformations.euler_from_quaternion(rot)
        q = tf2.transformations.quaternion_from_euler(0.0,0.0,e[2])
 
        self.br.sendTransform(trans, q, rospy.Time.now(),"baselink_pred","world")
        mark_arr =  MarkerArray()
        mark_arr_list = []

        for j in range(10):

            pred_x1 = pred_pose[0][j]
            pred_y1 = pred_pose[1][j]

            l2 = pred_x1**2 + pred_y1**2
            if(l2 < 0.1):
                l2 = 0.0
            if(pred_x1 < 0.0):
                pred_x1 = 0.0

            r = l2 / (2*pred_y1)

            for i in range(10):
                mark = Marker()
                mark.header.frame_id = "world"
                mark.header.stamp = rospy.Time.now()
                mark.type = 0

                mark.ns = "basic_shapes"
                mark.id = i + 10*j

                mark.action = Marker.ADD

                mark.color.r = 0.0
                mark.color.g = 1.0
                mark.color.b = 0.0
                mark.color.a = 1.0

                mark.scale.x = 0.2
                mark.scale.y = 0.2
                mark.scale.z = 3.0

                mark.points.append(Point())

                # linear
                #pred_x = pred_x1*(i+1)/4
                #pred_y = pred_y1*(i+1)/4

                pred_x = pred_x1*(i+1)/10
                pred_y = r - np.sqrt(abs(r**2 - pred_x**2))
                if(pred_y1 < 0.0 ):
                    pred_y = r + np.sqrt(abs(r**2 - pred_x**2))

                yaw = e[2]
                pred_x_base = np.cos(yaw)*pred_x - np.sin(yaw)*pred_y
                pred_y_base = np.sin(yaw)*pred_x + np.cos(yaw)*pred_y
                mark.points[0].x = trans[0] + pred_x_base
                mark.points[0].y = trans[1] + pred_y_base 
                mark.points[0].z = trans[2]+1.0

                mark.points.append(Point())
                mark.points[1].x = trans[0]+ pred_x_base
                mark.points[1].y = trans[1]+ pred_y_base 
                mark.points[1].z = trans[2]

                mark.lifetime = rospy.Duration()
                mark_arr_list.append(mark)

        mark_arr.markers = mark_arr_list
        self.pub_res.publish(mark_arr)

    def main(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('rostensorflow')
    tensor = RosTensorFlow()
    tensor.main()
