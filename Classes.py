import matplotlib
matplotlib.use('TkAgg')
import gym
from gym import spaces
import numpy as np
from commons import *
import matplotlib.pyplot as plt
class RLTask(AbstractRLTask):

    def interact(self, n_episodes):
        """
            This function executes n_episodes of interaction between the agent and the environment.

            :param n_episodes: the number of episodes of the interaction
            :return: a list of episode avergae returns  (see assignment for a definition
        """
        avg_returns = []
        for i in range(n_episodes):
            total_return = 0
            episode_end = False
            observation = self.env.reset()
            reward = 0
            num_actions = 0
            while not episode_end:
                action = self.agent.act(observation, reward)
                observation, reward, episode_end, info = self.env.step(action)
                total_return += reward
                num_actions+=1

            avg_returns.append(total_return/(num_actions+1))

        return avg_returns

    def visualize_episode(self, max_number_steps = None):

        observation = self.env.reset()
        episode_end = False
        step_count = 0
        while not episode_end:
            action = self.agent.act(observation, 0)
            observation, reward, episode_end, info = self.env.step(action)
            self.env.render()
            step_count+=1
            if max_number_steps is not None and step_count == max_number_steps:
                break

class RandomAgent(AbstractAgent):
    # def __init__(self):
    #     self.num_actions = 4
    #     super().__init__(self, 0, spaces.Discrete(self.num_actions))

    def act(self, state, reward=0):
        return self.action_space.sample()

    def onEpisodeEnd(self, reward, episode):
        print("yay")


class GoalFindingEnv(gym.Env):
    def __init__(self, x, y):
        # super().__init__()
        self.action_space = spaces.Discrete(4)  # Up Down Left Right
        self.observation_space = spaces.Discrete(x * y)  #
        self.x_size = x
        self.y_size = y
        self.x_goal = x - 1
        self.y_goal = y - 1
        self.reward = -1
        self.reset()

    def get_action_space(self):
        return self.action_space

    def step(self, action):
        self.move_agent(action)
        reached_goal = self.x_agent == self.x_goal and self.y_agent == self.y_goal
        return (self.x_agent, self.y_agent), self.reward, reached_goal, {}

    def reset(self):
        self.x_agent = 0
        self.y_agent = 0

        self.grid = np.zeros((self.x_size, self.y_size))  # Reset the grid
        self.grid[self.x_agent, self.y_agent] = 2
        self.grid[self.x_goal, self.y_goal] = 1

        return self.x_agent, self.y_agent


    def move_agent(self, action):

        self.grid[self.x_agent, self.y_agent] = 0

        if action == 0:  # up
            self.y_agent = max(self.y_agent - 1, 0)
        elif action == 1:  # down
            self.y_agent = min(self.y_agent + 1, self.y_size - 1)
        elif action == 2:  # left
            self.x_agent = max(self.x_agent - 1, 0)
        elif action == 3:  # right
            self.x_agent = min(self.x_agent + 1, self.x_size - 1)

        self.grid[self.x_agent, self.y_agent] = 2

    def render(self):
        for i in range(self.y_size):
            print(" _" * self.x_size)
            print("|", end="")
            for j in range(self.x_size):
                if self.grid[j, i] == 0:
                    print(" ", end="")
                elif self.grid[j, i] == 1:
                    print("G", end="")
                elif self.grid[j, i] == 2:
                    print("A", end="")
                print("|", end="")
            print("")
        print(" _" * self.x_size)
        print("\n\n")

    def close(self):
        pass

custom_env = GoalFindingEnv(5,5)
custom_agent = RandomAgent(0, custom_env.get_action_space())
rl_task = RLTask(custom_env, custom_agent)
avg_returns = rl_task.interact(10000)
plt.plot(range(0, 10000), avg_returns)
rl_task.visualize_episode()
plt.show()
