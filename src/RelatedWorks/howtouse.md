# How to Use

CARLA simulator

```
./CarlaUE4.sh -benchmark -fps=10 -quality-level=Epic

docker start 8529f939ee5d
docker attach 8529f939ee5d

cd CIA/ChauffeurNet/RelatedWorks/
source ../../../lgsvl/Autoware/ros/install/setup.bash

python tensorflow_in_ros_chau_ours.py

```

team_code/Autoware/ros/install/carla_ackermann_control/lib/carla_ackermann_control/carla_ackermann_control_node.py



## Server with GPU


srun -p hpc -w aventador --gres=gpu:1   -N 1 -n 1  --pty bash

ここでサーバーをつけると画面は見えないが起動はできるので、↑をやった後、dockerに-itで入って、tmuxしてサーバー起動しつつ実行すれば評価できる。↓

agent用

docker run --rm -v /home/s_seiya/workspace2:/home --net=host --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -it carla_tester:latest /bin/bash  


server用
docker run --rm -v /home/s_seiya/workspace2:/home -it --net=host --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES carla-overnight:latest /bin/bash

↑は基本的に別々でサーバー建てないといけない。carla-overnightは環境は良いけどAPIが違うから注意。
srun->tmux->$CUDA_VISIBLE_DEVICESを継承->docker両建てすればGPU1台体制も可能。しかし、netはどのみちhostingしているので複数台同時評価するにはちょっと大変。両方のdockerを含む親dockerが必要になってくる。
