
# How to Install

## Deepware modules

Deepware modules are for data collection, model training and model evaluation.

```bash
git clone https://github.com/shunchan0677/deepware
```

Install virtual env and Setup development environment(ex. python 2.7.5)

```bash
pip install virtualenv 

mkdir virtualenv
cd virtualenv
virtualenv deepware

cd deepware
source bin/activate
```

Install requirements module

```bash
pip install requirements.txt
```

## For Level5 dataset

Install https://github.com/lyft/nuscenes-devkit  
you can try to pip install. if it's failed, you can git clone this repo and set deepware/src

And Download dataset from https://level-5.global/data/


## For Automatic data collector

Automatic data collector is simulator environment used CARLA, Docker and Autoware. The collected data is saved as ROSBAG.

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



