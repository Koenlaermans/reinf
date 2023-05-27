import matplotlib
matplotlib.use('TkAgg')
import gym
import minihack_env as me
from gym import spaces
import numpy as np
from commons import *
import matplotlib.pyplot as plt
from numpy import random
import time
import copy

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

class RLTaskLearning(AbstractRLTask):

    def interact(self, n_episodes):
        """
            This function executes n_episodes of interaction between the agent and the environment.

            :param n_episodes: the number of episodes of the interaction
            :return: a list of episode avergae returns  (see assignment for a definition
        """
        avg_returns = []
        returns = []
        observation = self.env.reset()
        self.agent.initialize(observation)
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
                num_actions += 1

            self.agent.onEpisodeEnd(reward, i)
            returns.append(total_return)
            avg_returns.append(sum(returns) / len(returns))
            print(i, num_actions, total_return, avg_returns[len(avg_returns)-1])
        return avg_returns

def visualize_episode(self, max_number_steps = None):

    observation = self.env.reset()
    episode_end = False
    step_count = 0
    self.env.render()
    while not episode_end:
        action = self.agent.act(copy.deepcopy(self.env), observation, 0)
        observation, reward, episode_end, info = self.env.step(action)
        self.env.render()
        step_count+=1
        if max_number_steps is not None and step_count == max_number_steps:
            break

class TemporalDifferenceAgent(AbstractAgent):

    def __init__(self, id, action_space):
        AbstractAgent.__init__(self, id, action_space)
        self.epsilon = 1
        self.discount = 1
        self.alpha = 0.1
        self.env_dict = {}

    def add_to_dict(self, encoding):
        if encoding in self.env_dict:
            return self.env_dict[encoding]
        else:
            self.env_dict[encoding] = len(self.env_dict.keys())
            return self.env_dict[encoding]

    def initialize(self, obs):

        state_size = calc_environment_size(get_crop_chars_from_observation(obs))

        print("State size is "+str(state_size))

        action_size = self.action_space.n

        self.q_function = np.zeros((state_size, action_size))

        self.prev_action = -1

    def get_q(self):
        return self.q_function

    def run(self, state):

        x = random.rand()
        if x<self.epsilon:
            chosen_action = self.action_space.sample()
        else:
            max_action_value = self.q_function[state][np.argmax(self.q_function[state])]
            max_actions = []
            for j in range(len(self.q_function[state])):
                if self.q_function[state][j] == max_action_value:
                    max_actions.append(j)
            chosen_action = random.choice(max_actions)

        return chosen_action

    def act(self, state, reward=0):
        encoded_state = self.add_to_dict(enc_obs(state))

        chosen_action = self.run(encoded_state)

        if self.prev_action != -1:

            error = reward + self.discount * self.q_function[encoded_state, chosen_action] - self.q_function[self.prev_state, self.prev_action]
            self.q_function[self.prev_state, self.prev_action] = self.q_function[self.prev_state, self.prev_action]+self.alpha*error

        self.prev_action = chosen_action
        self.prev_state = encoded_state
        return chosen_action

    def onEpisodeEnd(self, reward, episode):
        self.epsilon = 0.99*self.epsilon
        print(self.epsilon)
        error = reward - self.q_function[self.prev_state, self.prev_action]
        self.q_function[self.prev_state, self.prev_action] = self.q_function[self.prev_state, self.prev_action] + self.alpha * error
class MonteCarloOnPolicyAgent(AbstractAgent):

    def __init__(self, id, action_space):
        AbstractAgent.__init__(self, id, action_space)
        self.epsilon = 1
        self.discount = 1
        self.env_dict = {}

    def add_to_dict(self, encoding):
        if encoding in self.env_dict:
            return self.env_dict[encoding]
        else:
            self.env_dict[encoding] = len(self.env_dict.keys())
            return self.env_dict[encoding]

    def initialize(self, obs):

        state_size = calc_environment_size(get_crop_chars_from_observation(obs))

        print("State size is "+str(state_size))

        action_size = self.action_space.n

        self.q_function = np.zeros((state_size, action_size))
        self.q_average_returns = np.zeros((state_size, action_size, 2))

        self.episode_trajectories = []
        self.prev_action = -1

    def get_q(self):
        return self.q_function

    def run(self, state):

        x = random.rand()
        if x<self.epsilon:
            chosen_action = self.action_space.sample()
        else:
            max_action_value = self.q_function[state][np.argmax(self.q_function[state])]
            max_actions = []
            for j in range(len(self.q_function[state])):
                if self.q_function[state][j] == max_action_value:
                    max_actions.append(j)
            chosen_action = random.choice(max_actions)

        return chosen_action


    def mc_first_visit(self):
        visited_state_action_pairs = []
        G = 0
        episode = self.episode_trajectories

        for t in range(len(episode)-1, -1, -1):
            state, action, reward = episode[t]
            G = self.discount*G + reward
            if (state, action) not in visited_state_action_pairs:
                visited_state_action_pairs.append((state, action))
                self.update_q(state, action, G)

    def update_q(self, state, action, G):
        state_action_pair = self.q_average_returns[state][action]
        state_action_pair[0] = (state_action_pair[0]*state_action_pair[1]+G)/(state_action_pair[1]+1)
        state_action_pair[1] += 1
        self.q_function[state][action] = state_action_pair[0]


    def act(self, state, reward=0):
        encoded_state = self.add_to_dict(enc_obs(state))

        if self.prev_action != -1:
            self.episode_trajectories.append((self.prev_state, self.prev_action, reward))

        chosen_action = self.run(encoded_state)
        self.prev_action = chosen_action
        self.prev_state = encoded_state
        return chosen_action

    def onEpisodeEnd(self, reward, episode):
        self.episode_trajectories.append((self.prev_state, self.prev_action, reward))
        self.mc_first_visit()
        self.episode_trajectories = []
        self.prev_action = -1
        self.epsilon = max(0.99*self.epsilon, 0.05)

        print(self.epsilon)


class FixedAgent(AbstractAgent):
    def act(self, state, reward=0):
        char_state = decode_ascii_array(get_crop_chars_from_observation(state))
        agent_x, agent_y = get_agent_pos(char_state)
        if out_of_bounds(agent_x+1, agent_y, char_state) or \
                is_wall(char_state[agent_x+1, agent_y]):
            return Action.RIGHT.value
        else:
            return Action.DOWN.value

    def onEpisodeEnd(self, reward, episode):
        print("yay")

class RandomAgent(AbstractAgent):

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

#Task 1.1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# custom_env = GoalFindingEnv(5,5)
# custom_agent = RandomAgent(0, custom_env.action_space)
# rl_task = RLTask(custom_env, custom_agent)
# avg_returns = rl_task.interact(10000)
# plt.plot(range(0, 10000), avg_returns)
# rl_task.visualize_episode()
# plt.show()

#Task 1.2 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# id = me.EMPTY_ROOM
# env = me.get_minihack_envirnment(id)
# custom_agent = FixedAgent(0, env.action_space)
# rl_task = RLTask(env, custom_agent)
# rl_task.visualize_episode(10)

#Task 2.1 a @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
id = me.ROOM_WITH_LAVA_MODIFIED
env = me.get_minihack_envirnment(id, max_episode_steps=10000)
# state = env.reset()
# plt.imshow(get_crop_pixel_from_observation(state))
# plt.show()
# exit(0)
custom_agent = MonteCarloOnPolicyAgent(0, env.action_space)
rl_task = RLTaskLearning(env, custom_agent)
avg_returns = rl_task.interact(5000)
plt.plot(range(0, 5000), avg_returns)
plt.show()