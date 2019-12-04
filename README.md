# Deepware : deep learning-based autonomous driving toolkit

Deepware is the Deep Learning-based Autonomous Driving Toolkit, mainly focuses on end-to-end and mid-to-mid driving. Deepware is used ROS/ROS2, Tensorflow, Autoware, Docker and CARLA/LGSVL. Deepware is provided automatic data collector, data extractor, data augmenter, model trainer and driving agents.


* [Automatic data collector](#Automatic-data-collector)
* [Data extractor](#Data-extractor)
* [Data augmenter](#Data-augmenter)
* [Model trainer](#Model-trainer)
* [Driving agents](#Driving-agents)


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



### 4. Collect data

## Data extractor

TODO

## Data augmenter

TODO

## Model trainer

TODO

## Driving agents

TODO
