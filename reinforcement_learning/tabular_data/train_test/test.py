import time
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from environments.tabularenv_test import TabularEnv


selected = "DQN"
option = "one"
model_path = f"models/one_attack/{selected}_{option}/{selected}_{option}.zip"

env = TabularEnv()
env.reset()

model = DQN.load(model_path,env=env)

#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1000)

attempts, correct = 0,0

vec_env = model.get_env()
obs = vec_env.reset()
inicio = time.time()
terminated = False
# while (not terminated):
for _ in range(18000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    attempts += 1
    if rewards > 0:
        correct += 1
    terminated = info[0]["terminated"]

fim = time.time()
accuracy = correct/attempts
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {info[0]['precision']:.2f}")
print(f"Recall: {info[0]['recall']:.2f}")
print(f"tempo de teste: {fim-inicio:.2f}")

env.close()