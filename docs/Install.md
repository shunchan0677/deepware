
# How to Install

## Deepware modules

Deepware modules are for data collection, model training and model evaluation.

```bash
git clone https://github.com/shunchan0677/deepware
```

Setup development environment

```bash
source activate
```

Install requirements module

```bash
pip install requirements.txt
```


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



