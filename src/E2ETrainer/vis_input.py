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
        self.list_image = np.zeros((1,13,256,256,3))
        self.pub_res = rospy.Publisher('result_mark', MarkerArray, queue_size=1)
        self.listener = tf2.TransformListener()
        self.br = tf2.TransformBroadcaster()

    def callback_ego_vehicle(self,image):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image, "bgr8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(cv_image, (256,256))
        self.list_image[:,10,:,:,:] = gray.reshape((1,256,256,3))/ 255.0
        print("ego_vehicle")


    def callback_hd_map(self,image):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image, "bgr8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(cv_image, (256,256))
        self.list_image[:,11,:,:,:] = gray.reshape((1,256,256,3))/ 255.0
        #print("hd_map")

    def callback_waypoint(self,image):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image, "bgr8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(cv_image, (256,256))
        self.list_image[:,12,:,:,:] = gray.reshape((1,256,256,3))/ 255.0
        #print("waypoint")

    def callback_occupancy_grid_0(self,image):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image, "bgr8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(cv_image, (256,256))
        self.list_image[:,0,:,:,:] = gray.reshape((1,256,256,3))/ 255.0
        #print("occupancy_grid_0")

    def callback_occupancy_grid_1(self,image):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image, "bgr8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(cv_image, (256,256))
        self.list_image[:,1,:,:,:] = gray.reshape((1,256,256,3))/ 255.0
        #print("occupancy_grid_1")

    def callback_occupancy_grid_2(self,image):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image, "bgr8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(cv_image, (256,256))
        self.list_image[:,2,:,:,:] = gray.reshape((1,256,256,3))/ 255.0
        #print("occupancy_grid_2")

    def callback_occupancy_grid_3(self,image):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image, "bgr8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(cv_image, (256,256))
        self.list_image[:,3,:,:,:] = gray.reshape((1,256,256,3))/ 255.0
        #print("occupancy_grid_3")

    def callback_occupancy_grid_4(self,image):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image, "bgr8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(cv_image, (256,256))
        self.list_image[:,4,:,:,:] = gray.reshape((1,256,256,3))/ 255.0
        #print("occupancy_grid_4")
        self.callback_model()

    def callback_model(self):
        print("save image")
        for i in range(13):
            cv2.imwrite("input/input"+str(i)+".jpg",np.reshape(self.list_image[:,i,:,:,:],(256,256,3))*255.0)
        

    def main(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('rostensorflow')
    tensor = RosTensorFlow()
    tensor.main()
