import gymnasium as gym
from stable_baselines3 import A2C, PPO, DQN
import os
from environments.tabularenv import TabularEnv
import time

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


TIMESTEPS = 200000

inicio = time.time()
model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=modelstr)
fim = time.time()

print(f'Tempo de treinamento: {fim-inicio:.2f}')


model.save(f"{models_dir}/{selected}_{option}")   
