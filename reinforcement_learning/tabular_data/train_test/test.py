from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from environments.tabularenv_val import TabularEnv


selected = "DQN"
option = "separated"
model_path = f"models/ordered/{selected}_{option}/{selected}_{option}.zip"

env = TabularEnv()
env.reset()

model = DQN.load(model_path,env=env)

#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1000)

attempts, correct = 0,0

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(680000):
    print(f"Iteration: {i}")
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    attempts += 1
    if rewards > 0:
        correct += 1
accuracy = correct/attempts
print(f'Accuracy reward: {accuracy:.2f}')
print(f'Precision: {info[0]:.2f}')
print(f'Recall: {info[1]:.2f}')
env.close()