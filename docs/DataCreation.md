
## Automatic data collector

Automatic data collector is used CARLA, Docker and Autoware. The collected data is saved as ROSBAG.

### 0. Requirements

* Ubuntu 16.04
* ROS Kinetic
* GPU setting using NVIDIA Driver

### 1. Install Docker

Following the docs, Please install docker system.  
https://docs.docker.com/install/linux/docker-ce/ubuntu/  

Following the code, user is added docker group not to need "sudo".

```bash
sudo groupadd docker
sudo gpasswd -a $USER docker
```

After rebooting PC, please check the response of following code.

```bash
docker images
```

If the installation is successed, the response is like this.

```bash
REPOSITORY          TAG                 IMAGE ID            CREATED             VIRTUAL SIZE
```



### 2. Download and extract CARLA binary

Following the google drive, please download the binary version of CARLA 0.9.5.  
https://drive.google.com/file/d/13QqmXtE0q6imMTmGidWtytJQp1fcPGtI/view?usp=sharing

After downloading, you can get "carla_095.tar.gz". Please extract the gz file in your home dir.


### 3. Pull Docker images of Automatic data collector 

Pull the docker images of Automatic data collector using the code.

```bash
docker pull shunchan0677/carla-data-collector
```


### 4. Start Collecting data

For collecting rosbag data, Running CARLA server and the docker image are required.

The code is running CARLA server.

```bash
cd CARLA_Latest
./CarlaUE4.sh -benchmark -fps=20 -quality-level=Epic
```

The code is running the docker image in another terminal.
```bash
docker run -it --net=host --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -e CHALLENGE_PHASE_CODENAME=debug_track_0 -v /media/<user>/<savedir>:/mnt shunchan0677/carla-data-collector:latest /bin/bash

#if you want to use object maps
#docker run -it --net=host --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -e CHALLENGE_PHASE_CODENAME=debug_track_0 -v /media/<user>/<savedir>:/mnt 20191210icra-all:latest /bin/bash

bash scenario_runner/srunner/challenge/run_evaluator.sh # in docker terminal
```

The rosbag data is saved on `<user>/<savedir>/PioMeidai`. (You need to check the path!)
If you want to select save topics and path, you should change the line 30 of "/workspace/team_code/carla-autoware/autoware_launch/points_raw.launch" in docker container.

### 5. Collected rosbag information

* tf
* image_raw
* points_raw
* vector_map
* lane_waypoints_array
* carla/hero/objects
* carla/hero/odometry
* occupancy_grid_0〜9
* vector_image_raw 
* vector_image_raw/ego_vehicle
* vector_image_raw/hd_map
* vector_image_raw/objects
* vector_image_raw/points
* vector_image_raw/waypoint
* vector_image_raw/without_ego_vehicle
* all

### 6. Sample Video

[![Sample Video](http://img.youtube.com/vi/YM7BAHmJwjM/0.jpg)](http://www.youtube.com/watch?v=YM7BAHmJwjM)
