import pybullet_envs
import gym
env = gym.make('HalfCheetahBulletEnv-v0')
env.render(mode="human")
for i_episode in range(100):
    observation = env.reset()
    for t in range(1000):
        still_open =env.render("human")
        #print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

print("State space dimension is:", env.observation_space.shape[0])
print("State upper bounds:", env.observation_space.high)
print("State lower bounds:", env.observation_space.low)
#print("Number of actions is:", env.action_space.n)
