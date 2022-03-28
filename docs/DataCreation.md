
###  Start Collecting data using Automatic Data Collector

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

### Collected rosbag information

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
