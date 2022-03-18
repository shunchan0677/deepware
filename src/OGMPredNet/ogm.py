#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import glob

def im2ogm(image):
    x_offset = 256.0*1.5/5.0
    y_offset = 0
    index_mat = np.zeros(image.shape)
    dis_mat = np.ones(image.shape)*10000000
    output = np.zeros(image.shape)
    a_mat = np.ones(image.shape[0])
    b_mat = np.asarray(range(image.shape[0]))

    x_mat = image.shape[0]/2 - np.outer(a_mat, b_mat.T) + x_offset
    y_mat = image.shape[1]/2 - np.outer(b_mat, a_mat.T) + y_offset

    dis_mat1 = x_mat**2 + y_mat**2
    dis_mat = dis_mat1.T
    tan_mat = np.arctan(y_mat/x_mat)
    tan_mat = np.where((x_mat == 0) * (y_mat >= 0), np.pi/2.0, tan_mat)
    tan_mat = np.where((x_mat == 0) * (y_mat < 0), - np.pi/2.0, tan_mat)
    tan_mat = np.where((x_mat < 0), tan_mat + np.pi, tan_mat)
    tan_mat = tan_mat + np.pi/2.0
    tan_mat = tan_mat*360 / (2*np.pi)
    index_mat1 = tan_mat.astype(np.int32)
    index_mat =  index_mat1.T

    dis_mat_im = np.where(image < 0.8, 100000000, dis_mat)

    for index in np.unique(index_mat):
        min_dis = np.min(np.where(index_mat != index, 100000000, dis_mat_im))
        output = np.where((index_mat == index) * (dis_mat < min_dis), 1, output)
    return output

"""

image_file = glob.glob("/media/brainiv/PioMeidai/CIA_dataset/ouput_2019-07-06-03-05-06/images/*")
image_file.sort()
print image_file[1000]

image_files = np.asarray(cv2.imread(image_file[1000]+"/occupancy_grid_0_.jpg", cv2.IMREAD_GRAYSCALE))/255

cv2.imshow("image",im2ogm(image_files)*255)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("image",image_files*255)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
