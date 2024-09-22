import gymnasium as gym
from stable_baselines3 import A2C, PPO, DQN
import os
from environments.tabularenv import TabularEnv


selected = "DQN"
option = "separated"
modelstr = f"{selected}_{option}"

models_dir = f"models/ordered/{modelstr}" 
logdir  = "logs/ordered"


if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

env = TabularEnv()
env.reset()

model = DQN("MlpPolicy", env,tensorboard_log=logdir)

TIMESTEPS = 680000

model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=modelstr)
model.save(f"{models_dir}/{selected}_{option}")   
