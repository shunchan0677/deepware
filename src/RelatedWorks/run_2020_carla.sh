#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda-9.0
export PYTHONPATH=/usr/bin/python:/work2/s_seiya/ski
export CUDA_HOME=/usr/local/cuda-9.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64


#python=/home/s_seiya/workspace2/virtualenv/millionx/bin/python

python -u train_e2e_2d_chauffeur_carla_2020.py


