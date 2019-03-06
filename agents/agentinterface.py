import gym
from utils import Logger
import time
import numpy as np


class Agent:
    def __init__(self, env: gym.Env, logger=Logger()):
        self.env = env
        self.exploration = True
        self.log = []  # episode log
        self.logger = logger  # external logger

    def act(self, state, explore):
        raise NotImplementedError

    def step_update(self, state, action, new_state, reward):
        # updates knowledge each time an action is taken
        pass

    def episode_update(self):
        # update inner knowledge based on log
        pass

    def episode(self, state, explore=True, visualize=False, max_steps=1000):
        total_reward = 0
        done = False
        c = 0
        if visualize:
            self.env.render()
        while not done and c<max_steps:
            action = self.act(state, explore)
            new_state, reward, done, info = self.env.step(action)
            if not done:
                self.step_update(state, action, new_state, reward, done)
            else:
                self.step_update(state, action, None, reward, done)
            self.log.append(dict(state=state, action=action, reward=reward))
            state = new_state
            total_reward += reward
            if visualize:
                self.env.render()
            c+=1
        return total_reward

    def train(self, nb_episodes=1000, visualize=False, verbose=None):
        t0 = time.time()
        for i in range(nb_episodes):
            t1 = time.time()
            reward = self.episode(self.env.reset(), visualize=visualize)
            self.episode_update()
            self.logger.log('Rewards', reward)
        print('Mean Episodic Time :{0}s'.format((time.time()-t0)/nb_episodes))


    def test_render(self, nb_episodes=100, visualize=True):
        rewards = []
        if visualize:
            self.env.render()
        for _ in range(nb_episodes):
            rewards.append(self.episode(self.env.reset(), explore=False, visualize=visualize))
        print('All Rewards :', rewards)
        return rewards
    
class KerasRLAgent(Agent):
    def __init__(self, env: gym.Env, logger=Logger()):
        super().__init__(env, logger)
    
    def train(self, nb_episodes=1000, visualize=False, verbose=1, nb_max_episode_steps=None):
        history = self.agent.fit(self.env, nb_steps=nb_episodes, visualize=visualize, verbose=verbose, nb_max_episode_steps=nb_max_episode_steps, log_interval=10000)
        return history
        
    def train_env(self, env,  nb_episodes=1000, visualize=False, verbose=1, nb_max_episode_steps=None):
        history = self.agent.fit(env, nb_steps=nb_episodes, visualize=visualize, verbose=verbose, nb_max_episode_steps=nb_max_episode_steps, log_interval=10000)
        return history
        
    def test(self, nb_episodes, visualize=True, verbose=1):
        if visualize:
            self.env.render()
        history = self.agent.test(self.env, nb_episodes=nb_episodes, visualize=visualize, verbose=verbose)
        print('Mean', np.mean(history.history['episode_reward']))
        print('Std', np.std(history.history['episode_reward']))
        return history
        
    def test_env(self, env, nb_episodes, visualize=True, verbose=1):
        t=time.time()
        history = self.agent.test(env, nb_episodes = nb_episodes, visualize=visualize, verbose=verbose)
        print('Mean', np.mean(history.history['episode_reward']))
        print('Std', np.std(history.history['episode_reward']))
        print(time.time()-t)
        return history
    
    def act(self, state, explore=True):
        action = self.agent.forward(state)
        return action
    
    def step_update(self, state, action, new_state, reward, done=False):
        self.agent.backward(reward, done)
        if new_state is None:
            self.agent.forward(state)
            self.agent.backward(0., terminal=False)
        return None
    
    def episode_update(self):
        self.log = []
        self.agent.random_process.sigma *= self.sigma_decay