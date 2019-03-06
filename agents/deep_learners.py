from agents import Agent, KerasRLAgent
from utils import Logger

import gym
import numpy as np
import time

from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent, DQNAgent, NAFAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.policy import BoltzmannQPolicy
from rl.processors import WhiteningNormalizerProcessor
from rl.core import Agent
from rl.util import *

class MujocoProcessor(WhiteningNormalizerProcessor):
    def process_action(self, action):
        return np.clip(action, -1., 1.)

class DDPG(KerasRLAgent):
    def __init__(self,
                 env : gym.Env,
                 logger=Logger(),
                 n_layers_actor=3,
                 n_units_actor=16,
                 n_layers_critic=3,
                 n_units_critic=32,
                 sigma_decay=1,
                 sigma=0.3
                 ):
        nb_actions = env.action_space.shape[0]
        
        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        for i in range(n_layers_actor):
            actor.add(Dense(n_units_actor))
            #actor.add(BatchNormalization())
            actor.add(Activation('relu'))
            #actor.add(LeakyReLU())
        actor.add(Dense(nb_actions))
        actor.add(Activation('tanh'))
        
        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = Concatenate()([action_input, flattened_observation])
        for i in range(n_layers_critic):
            x = Dense(n_units_critic)(x)
            #x = BatchNormalization()(x)
            x = Activation('relu')(x)
            #x = LeakyReLU()(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)
        
        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
        flattened_observation = Flatten()(observation_input)
        xo = Dense(n_units_critic, activation='relu')(flattened_observation)
        #xo = Dense(n_units_critic, activation='relu')(xo)
        x = Concatenate()([xo, action_input])
        for i in range(n_layers_critic-1):
            x = Dense(n_units_critic, activation='relu')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)
        
        memory = SequentialMemory(limit=1000000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=sigma)
        agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                          memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                          random_process=random_process, gamma=.99, target_model_update=1e-3, batch_size=64, train_interval=4)
        agent.compile([Adam(lr=.0001, clipnorm=1.), Adam(lr=.0001)], metrics=['mae'])
        self.agent = agent
        self.env = env
        self.sigma_decay = sigma_decay
        super().__init__(env, logger)
        
    

class NAF(KerasRLAgent):
    def __init__(self, 
                 env : gym.Env,
                 logger=Logger()):
        nb_actions = env.action_space.shape[0]
        
#        V_model = Sequential()
#        V_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
#        V_model.add(Dense(16))
#        V_model.add(Activation('relu'))
#        V_model.add(Dense(16))
#        V_model.add(Activation('relu'))
#        V_model.add(Dense(16))
#        V_model.add(Activation('relu'))
#        V_model.add(Dense(1))
#        V_model.add(Activation('linear'))
#        
#        mu_model = Sequential()
#        mu_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
#        mu_model.add(Dense(16))
#        mu_model.add(Activation('relu'))
#        mu_model.add(Dense(16))
#        mu_model.add(Activation('relu'))
#        mu_model.add(Dense(16))
#        mu_model.add(Activation('relu'))
#        mu_model.add(Dense(nb_actions))
#        mu_model.add(Activation('linear'))
#        
#        action_input = Input(shape=(nb_actions,), name='action_input')
#        observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
#        x = Concatenate()([action_input, Flatten()(observation_input)])
#        x = Dense(32)(x)
#        x = Activation('relu')(x)
#        x = Dense(32)(x)
#        x = Activation('relu')(x)
#        x = Dense(32)(x)
#        x = Activation('relu')(x)
#        x = Dense(((nb_actions * nb_actions + nb_actions) // 2))(x)
#        x = Activation('linear')(x)
#        L_model = Model(inputs=[action_input, observation_input], outputs=x)
        
        V_model = Sequential()
        V_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        V_model.add(Dense(16))
        V_model.add(Activation('relu'))
        V_model.add(Dense(16))
        V_model.add(Activation('relu'))
        V_model.add(Dense(16))
        V_model.add(Activation('relu'))
        V_model.add(Dense(1))
        V_model.add(Activation('linear'))
        print(V_model.summary())
        
        mu_model = Sequential()
        mu_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        mu_model.add(Dense(16))
        mu_model.add(Activation('relu'))
        mu_model.add(Dense(16))
        mu_model.add(Activation('relu'))
        mu_model.add(Dense(16))
        mu_model.add(Activation('relu'))
        mu_model.add(Dense(nb_actions))
        mu_model.add(Activation('linear'))
        print(mu_model.summary())
        
        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
        x = Concatenate()([action_input, Flatten()(observation_input)])
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(((nb_actions * nb_actions + nb_actions) // 2))(x)
        x = Activation('linear')(x)
        L_model = Model(inputs=[action_input, observation_input], outputs=x)
        print(L_model.summary())
        
        memory = SequentialMemory(limit=100000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.1, size=nb_actions)
        agent = NAFAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                 memory=memory, random_process=random_process)
        agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
        self.agent = agent
        self.env = env
        super().__init__(env, logger)
        
class DQN(KerasRLAgent):
    def __init__(self, 
                 env : gym.Env,
                 logger=Logger()):
        nb_actions = env.action_space.shape[0]
        
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        model.add(Activation('linear'))
        
        policy = BoltzmannQPolicy()
        memory = SequentialMemory(limit=100000, window_length=1)
        agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                       target_model_update=1e-2, policy=policy)
        agent.compile(Adam(lr=1e-3), metrics=['mae'])
        self.agent = agent
        self.env = env
        super().__init__(env, logger)
        
class MACEAgent(KerasRLAgent):
    def __init__(self, env : gym.Env):
        self.agent = MACE(env)
        self.env = env
        
class MACE(Agent):
    def __init__(self, 
                 env : gym.Env):
        nb_actions = env.action_space.shape[0]
        
        obs_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
        x = Dense(units=256, activation='relu')(obs_input)
        x_critic = Dense(units=128, activation='relu')(x)
        q_value = Dense(units=1)(x_critic)
        
        x_actor = Dense(units=128, activation='relu')(x)
        action = Dense(units=nb_actions)(x_actor)
        
        actor = Model(inputs=[obs_input], outputs = action) 
        critic = Model(inputs=[obs_input], outputs = q_value) 
        
        actor.compile(Adam(lr=1e-3), loss=['mse'])
        critic.compile(Adam(lr=1e-3), loss=['mse'])
        
        self.target_actor = clone_model(self.actor, self.custom_model_objects)
        self.target_actor.compile(optimizer='sgd', loss='mse')
        self.target_critic = clone_model(self.critic, self.custom_model_objects)
        self.target_critic.compile(optimizer='sgd', loss='mse')
        
        self.actor = actor
        self.critic = critic
        
        self.memory = SequentialMemory(limit=100000, window_length=1)
        self.memory_interval=1
        
        self.nb_steps_warmup_critic=1000
        self.nb_steps_warmup_actor=1000
        
        self.train_interval=4
        
        self.processor=None
        
    
    def process_state_batch(self, state):
        batch = self.process_state_batch([state])
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)
    
    def select_action(self, state):
        batch = [state]
        action = self.actor.predict_on_batch(batch).flatten()
        # Apply noise, if a random process is set.
        if self.training and self.random_process is not None:
            noise = self.random_process.sample()
            assert noise.shape == action.shape
            action += noise
        return action
    
    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        action = self.select_action(state)  # TODO: move this into policy

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action
    
    def backward(self, reward, terminal=False):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        can_train_either = self.step > self.nb_steps_warmup_critic or self.step > self.nb_steps_warmup_actor
        if can_train_either and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            action_batch = np.array(action_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert action_batch.shape == (self.batch_size, self.nb_actions)

            # Update actor and critic, if warm up is over.
            if self.step > self.nb_steps_warmup:
                if len(self.critic.inputs) >= 3:
                    state1_batch_with_action = state1_batch[:]
                else:
                    state1_batch_with_action = [state1_batch]
                target_q_values = self.target_critic.predict_on_batch(state1_batch_with_action).flatten()
                assert target_q_values.shape == (self.batch_size,)

                # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
                # but only for the affected output units (as given by action_batch).
                discounted_reward_batch = self.gamma * target_q_values
                discounted_reward_batch *= terminal1_batch
                assert discounted_reward_batch.shape == reward_batch.shape
                targets = (reward_batch + discounted_reward_batch).reshape(self.batch_size, 1)

                # Perform a single batch update on the critic network.
                if len(self.critic.inputs) >= 3:
                    state0_batch_with_action = state0_batch[:]
                else:
                    state0_batch_with_action = [state0_batch]
                #state0_batch_with_action.insert(self.critic_action_input_idx, action_batch)
                metrics = self.critic.train_on_batch(state0_batch_with_action, targets)
                if self.processor is not None:
                    metrics += self.processor.metrics

                #Actor
                target_q_values0 = self.target_critic.predict_on_batch(state0_batch_with_action).flatten()
                delta = targets - target_q_values0
                if len(self.actor.inputs) >= 2:
                    inputs = state0_batch[:]
                else:
                    inputs = [state0_batch]
                inputs = inputs[delta>0]
                actions_target = action_batch[delta>0]
                #state0_batch_with_action.insert(self.critic_action_input_idx, action_batch)
                self.actor.train_on_batch(inputs, actions_target)

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_models_hard()

        return metrics
