# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 17:23:01 2019

@author: tgill
"""
import gym
from gym.wrappers import Monitor
import pybullet_envs
import numpy as np

from agents.q_learners import QLearner
from agents.deep_learners import DDPG, NAF, DQN, MACEAgent, MACEDDPG
from utils import ShowVariableLogger
from envs import HalfCheetahEnv
import matplotlib.pyplot as plt

#ENV = gym.make('HalfCheetahBulletEnv-v0')
#ENV = gym.make('Walker2DBulletEnv-v0')
#ENV = gym.make('Walker2d-v2')
#ENV = gym.make('HalfCheetah-v2')
#ENV = gym.make('InvertedPendulum-v2')

ENV = HalfCheetahEnv()
from gym.wrappers.time_limit import TimeLimit
ENV = TimeLimit(ENV, max_episode_steps=1000)
ENV.render(mode="human")

#agent = QLearner(env=ENV,
#                 logger=ShowVariableLogger(average_window=100),
#                 boxes_resolution=2)
#
logger=ShowVariableLogger(average_window=100)

agent = DDPG(env=ENV,
                    logger=ShowVariableLogger(average_window=1),
                     n_layers_actor=2,
                     n_units_actor=300,
                     n_layers_critic=2,
                     n_units_critic=300,
                    )
#agent = MACEAgent(env=ENV)
#agent = MACEDDPG(env=ENV)
##agent = DQN(env=ENV,
##            logger=ShowVariableLogger(average_window=1))
#

#rewards = agent.test(nb_episodes=10, visualize=True)

tr_ep_rewards=[]
ep_rewards=[]

#.agent.agent.nb_steps_warmup=50000
for i in range(100):
    print("Iteration", i)
    hist_train = agent.train(nb_episodes=100000, visualize=False, verbose=1, nb_max_episode_steps=1000)
    #agent.agent.nb_steps_warmup=0
    tr_ep_rewards.append(np.mean(hist_train.history['episode_reward']))
    for rew in hist_train.history['episode_reward']:
        logger.log('Rewards', rew)
    #rewards = agent.test_render(nb_episodes=3, visualize=True)
    rewards = agent.test(nb_episodes=10, visualize=True)
    ep_rewards.append(np.mean(rewards.history['episode_reward']))
    #ENV.render(mode="human", close=True)

#print("Mean reward", np.mean(rewards.history['episode_reward']))