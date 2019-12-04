# Deepware : deep learning-based autonomous driving toolkit

Deepware is the Deep Learning-based Autonomous Driving Toolkit, mainly focuses on end-to-end and mid-to-mid driving. Deepware is used ROS/ROS2, Tensorflow, Autoware, Docker and CARLA/LGSVL. Deepware is provided automatic data collector, data extractor, data augmenter, model trainer and driving agents.

* [Related works](#Related-works)
* [Automatic data collector](#Automatic-data-collector)
* [Data extractor](#Data-extractor)
* [Data augmenter](#Data-augmenter)
* [Model trainer](#Model-trainer)
* [Driving agents](#Driving-agents)

## Related works
* Mariusz Bojarski, et al. "End to End Learning for Self-Driving Cars,"
  * https://arxiv.org/abs/1604.07316
* Felipe Codevilla, et al. "End-to-end Driving via Conditional Imitation Learning,"
  * https://arxiv.org/abs/1710.02410
* Shunya Seiya, et al. "End-to-End Navigation with Branch Turning Support Using Convolutional Neural Network,"
  * https://www.researchgate.net/publication/331866059_End-to-End_Navigation_with_Branch_Turning_Support_Using_Convolutional_Neural_Network
* Jeffrey Hawke, et al. "Urban Driving with Conditional Imitation Learning,"
  * https://arxiv.org/abs/1912.00177
* Mayank Bansal, et al. "ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst,"
  * https://arxiv.org/abs/1812.03079

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

Following the code, please download the binary version of CARLA 0.9.4.(https://carlachallenge.org/get-started/)

```bash
wget http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/Dev/CARLA_Latest.tar.gz
```

After downloading, you can get "CARLA_Latest.tar.gz". Please extract the gz file in your home dir.


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
docker run -it --net=host --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -e CHALLENGE_PHASE_CODENAME=debug_track_0 -v /mnt:/media/brainiv shunchan0677/carla-data-collector:latest /bin/bash

bash scenario_runner/srunner/challenge/run_evaluator.sh
```

## Data extractor

TODO

## Data augmenter

TODO

## Model trainer

TODO

## Driving agents

TODO
