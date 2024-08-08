import gymnasium as gym
import numpy as np
from gymnasium import spaces
import numpy as np
import random

class TabularEnv(gym.Env):
    """
    Action Space:
    - Discrete space with two actions (0 or 1). For Classification

    Observation Space:
    - Box space with shape (1, 18) and dtype float32, representing a set of features for the space ship control task.

    Methods:
    - step(action): Takes an action and returns the next observation, reward, done flag, and additional info.
    - reset(): Resets the environment to the initial state and returns the initial observation.
    - _next_obs(): Returns the next observation based on the current dataset and mode.

    Attributes:
    - action_space: Discrete space with two actions (0 or 1).
    - observation_space: Box space with shape (1, 18) and dtype float32.
    - row_per_episode (int): Number of rows per episode.
    - step_count (int): Counter for the number of steps within the current episode.
    - x, y: Features and labels from the dataset.
    - random (bool): If True, observations are selected randomly from the dataset; otherwise, follows a sequential order.
    - dataset_idx (int): Index to keep track of the current observation in sequential mode.
    - expected_action (int): Expected action based on the current observation.
    """

    def __init__(self, row_per_episode=1, dataset=(df_train_X, df_train_y), random=True):
        super().__init__()

        # Define action space
        self.action_space = gym.spaces.Discrete(2)

        # Define observation space
        obersevation = np.array([[np.finfo('float32').max] * 18], dtype=np.float32)
        self.observation_space = spaces.Box(-obersevation, obersevation, shape=(1, 18), dtype=np.float32)

        # Initialize parameters
        self.row_per_episode = row_per_episode
        self.step_count = 0
        self.x, self.y = dataset
        self.random = random
        self.dataset_idx = 0

    def step(self, action):
        """
        Takes an action and returns the next observation, reward, done flag, and additional info.

        Parameters:
        - action (int): The action taken by the agent.

        Returns:
        - obs (numpy array): The next observation.
        - reward (int): The reward obtained based on the action.
        - done (bool): Flag indicating whether the episode is done.
        - info (dict): Additional information.
        """

        done = False
        reward = int(action == self.expected_action)

        obs = self._next_obs()

        self.step_count += 1
        if self.step_count >= self.row_per_episode:
            done = True

        return obs, reward, done, {}

    def reset(self):
        """
        Resets the environment to the initial state and returns the initial observation.

        Returns:
        - obs (numpy array): The initial observation.
        """

        self.step_count = 0

        obs = self._next_obs()
        return obs

    def _next_obs(self):
        """
        Returns the next observation based on the current dataset and mode.

        Returns:
        - obs (numpy array): The next observation.
        """

        if self.random:
            next_obs_idx = random.randint(0, len(self.x) - 1)
            self.expected_action = int(self.y[next_obs_idx])
            obs = self.x[next_obs_idx]

        else:
            obs = self.x[self.dataset_idx]
            self.expected_action = int(self.y[self.dataset_idx])

            self.dataset_idx += 1
            if self.dataset_idx >= len(self.x):
                self.dataset_idx = 0

        return obs