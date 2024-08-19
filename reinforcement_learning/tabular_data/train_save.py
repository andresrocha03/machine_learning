import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

import os
from tabularenv import TabularEnv

modelstr = "DQN"
models_dir = "models/" + modelstr
logdir  = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

env = TabularEnv()
env.reset()

model = DQN("MlpPolicy", env, verbose=1,tensorboard_log=logdir)

TIMESTEPS = 10000

model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=modelstr)
model.save(f"{models_dir}/DQN_model")   
