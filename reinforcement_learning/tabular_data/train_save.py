import gymnasium as gym
from stable_baselines3 import PPO
import os
from tabularenv import TabularEnv

modelstr = "PPO"

models_dir = "models/" + modelstr
logdir  = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = TabularEnv()
env.reset()

model = PPO("MlpPolicy", env, verbose=1,tensorboard_log=logdir)

TIMESTEPS = 50000

model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=modelstr)
model.save(f"{models_dir}/{TIMESTEPS}")    

attempts, correct = 0,0
# obs = env.reset()

vec_env = model.get_env()
obs = vec_env.reset()
print(type(obs))
for i in range(12000):
    action, _states = model.predict(obs)
    #print(action)
    obs, rewards, dones, info = vec_env.step(action)
    attempts += 1
    if rewards > 0:
        correct += 1
    #obs = env.reset()
print('Accuracy: {0}%'.format((float(correct) / attempts) * 100))


# vec_env = model.get_env()
# obs = vec_env.reset()

# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = vec_env.step(action)
#     vec_env.render("human")
   


env.close()