#!/bin/bash
# J.Zhu, MPI-IS
# run training of the augmented DDPG agent for ETC in Pendulum env

# set param
n_epoch=500
REW_TYPE="const" # const, inv, linear
param_val=0.1 # this parameter control the trade-off between the communication freq and performance
env_name="Pendulum-v1"
seed=0 # use seed to reproduce results

# run script to train the resource aware agent
python -m baselines.ddpg.main_kai --reward_param_scaling ${param_val} --reward_param_type ${REW_TYPE} --env-id $env_name --nb-epochs ${n_epoch} --no-my_render --seed ${seed}
