import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import glob
import copy
import os
import cv2
import numpy as np

image_file = glob.glob("/home/s_seiya/tmp/CARLA_dataset/_2019-09-04-05-57-49/images/00000/*")
output = "/home/s_seiya/workspace2/rotate_images/"


cameraMatrix1 = np.loadtxt('cameraMatrix0.csv',delimiter = ',')
cameraMatrix2 = np.loadtxt('cameraMatrix0.csv',delimiter = ',')
distCoeffs2 = np.loadtxt('distCoeffs0.csv',delimiter = ',')

i = 0
for path in image_file:
    print(path)
    image = cv2.imread(path)
    if(path.split("/")[-1]=="image_raw_.jpg"):
        print("image_raw")
        image = cv2.resize(image,(1920,1440))
        newimageSize = (image.shape[1],image.shape[0])
        r=np.pi/180*5*(4-i)
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
        if(path.split("/")[-1]=="occupancy_grid_4_.jpg"):
            image[320-vehicle_x*400/80:320+vehicle_x*400/80 ,200-vehicle_y*400/80:200+vehicle_y*400/80] = 0
        trans = cv2.getRotationMatrix2D(center, angle , scale)
        image = cv2.warpAffine(image, trans, (400,400))
        if(path.split("/")[-1]=="occupancy_grid_4_.jpg"):
            vehicle_x = 6/2
            vehicle_y = 3/2
            image[320-vehicle_x*400/80:320+vehicle_x*400/80/2 ,200-vehicle_y*400/80:200+vehicle_y*400/80] = 255
            image[320-9:320-7 ,200-vehicle_y*400/80:200+vehicle_y*400/80] = 0
    cv2.imwrite(output+path.split("/")[-1],image)
