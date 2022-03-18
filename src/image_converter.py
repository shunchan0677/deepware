import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import glob
import copy
import os
import cv2
import numpy as np

output_dir = "/home/s_seiya/tmp/lyft_data"
filenames = glob.glob("/work5/s_seiya/level5dataset/v1.01-train/dataset/scene*/*")
filenames.sort()
for names in filenames:
    output_file = output_dir + "/" + names.split("/")[-2] + "/" + names.split("/")[-1]
    os.makedirs(output_file, exist_ok=True)
    print (names)
    cam = cv2.imread(names + "/image.png")
    cam = cam[200:880,200:1720,:]
    cam = cv2.resize(cam,(200,66))
    cv2.imwrite(output_file + "/image.png",cam)

    for i in range(10):
        box = cv2.imread(names + "/bbox_" + str(i) + ".png")
        box[:,:,1] = np.where(box[:,:,0] > 250, 0, 255)
        box[:,:,2] = np.zeros((400,400))#where(box[:,:,0] > 250, 0, 255)
        box[:,:,0] = box[:,:,2]
        box = cv2.resize(box,(257,257))
        box = box[:256,:256,:]
        
        cv2.imwrite(output_file + "/bbox_" + str(i) + ".png", box)

        lid = cv2.imread(names + "/lidar_" + str(i) + ".png")
        lid[:,:,1] = np.where(lid[:,:,0] > 250, 0, 255)
        lid[:,:,2] = np.where(lid[:,:,0] > 250, 0, 255)
        lid[:,:,0] = lid[:,:,2]
        lid = cv2.resize(lid,(257,257))
        lid = lid[:256,:256,:]

        cv2.imwrite(output_file + "/lidar_" + str(i) + ".png", lid)

    ego = cv2.imread(names + "/ego_pose.png")
    #ego = cv2.resize(ego,(257,257))
    ego[:,:,2] = np.where(ego[:,:,1] > 250, 0, ego[:,:,2])
    ego[:,:,0] = np.zeros((400,400))#where(ego[:,:,1] > 250, 0, 255)
    ego[:,:,1] = ego[:,:,0]
    ego = cv2.resize(ego,(257,257))
    ego = ego[:256,:256,:]
    cv2.imwrite(output_file + "/ego_pose.png",ego)

    way = cv2.imread(names + "/waypoint.png")
    #way = cv2.resize(way,(257,257))
    way[:,:,0] = np.where(way[:,:,1] > 250, 0, 255)
    way[:,:,1] = np.zeros((400,400))
    way[:,:,2] = way[:,:,1]
    way = cv2.resize(way,(257,257))
    way = way[:256,:256,:]
    cv2.imwrite(output_file + "/waypoint.png",way)


    map_im = cv2.imread(names + "/map.png")
    #map_im = cv2.resize(map_im,(257,257))
    map_con = copy.copy(map_im)
    map_con[:,:,0] = np.where((map_im[:,:,0] > 250 )*( map_im[:,:,1] > 250)*( map_im[:,:,2] > 250), 0, map_con[:,:,0])
    map_con[:,:,1] = np.where((map_im[:,:,0] > 250 )*( map_im[:,:,1] > 250)*( map_im[:,:,2] > 250), 0, map_con[:,:,1])
    map_con[:,:,2] = np.where((map_im[:,:,0] > 250 )*( map_im[:,:,1] > 250)*( map_im[:,:,2] > 250), 0, map_con[:,:,2])
    map_im = cv2.resize(map_con,(257,257))
    map_im = map_im[:256,:256,:]
    cv2.imwrite(output_file + "/map.png",map_im)

