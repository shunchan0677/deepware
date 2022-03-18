# ChauffeurNet

This repositry is implimentation of instant version of ChauffeurNet([https://arxiv.org/pdf/1903.00640.pdf]).

We choose the method because Waymo's paper([https://arxiv.org/pdf/1812.03079.pdf]) doesn't have descriptions of parameter of each features and models.


## Platform

We use ROS and Autoware.

## Evaluation

 - Offline Evaluation
   - Lyft dataset(https://level5.lyft.com/dataset/#data-collection)
   
 - Online Evaluation
   - CARLA Simulator(https://carlachallenge.org/)

## How to create training dataset

1. Correct driving dataset
 - Camera for traffic light recognition
 - Lidar for localization and OGMPred
 - GPS for localization
 
 If you can use docker hub, you can clone docker image. and run: 
 docker run -it --net=host --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -e CHALLENGE_PHASE_CODENAME=debug_track_0 -v /mnt:/media/brainiv  carla_ros_auto_data_collector:latest /bin/bash

2. Create Localization dataset
3. Create Feature map dataset
 - extract feature map rosbag
 - extract feature maps
 - "rosbag record /tf /occupancy_grid_0 /occupancy_grid_1 /occupancy_grid_2 /occupancy_grid_3 /occupancy_grid_4 /occupancy_grid_5 /occupancy_grid_6 /occupancy_grid_7 /occupancy_grid_8 /occupancy_grid_9 /vector_image_raw /vector_image_raw/ego_vehicle /vector_image_raw/hd_map /vector_image_raw/objects /vector_image_raw/points  /vector_image_raw/waypoint /vector_image_raw/without_ego_vehicle"

4. Augmented data

5. Training model



