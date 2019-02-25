# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 17:23:01 2019

@author: tgill
"""
import gym
import numpy as np

from agents.q_learners import QLearner
from agents.deep_learners import DDPG, NAF, DQN
from utils import ShowVariableLogger

ENV = gym.make('HalfCheetah-v2')

#agent = QLearner(env=ENV,
#                 logger=ShowVariableLogger(average_window=100),
#                 boxes_resolution=2)
#
agent = DDPG(env=ENV,
                    logger=ShowVariableLogger(average_window=1),
                     n_layers_actor=2,
                     n_units_actor=300,
                     n_layers_critic=2,
                     n_units_critic=300,
                    )


#agent = DQN(env=ENV,
#            logger=ShowVariableLogger(average_window=1))

for i in range(10):
    hist_train = agent.train(nb_episodes=100000, visualize=False, verbose=1, nb_max_episode_steps=1000)
    rewards = agent.test(nb_episodes=3, visualize=True)

print("Mean reward", np.mean(rewards.history['episode_reward']))