import gym
import time
from gym import envs
print([e.id for e in envs.registry.all()])

env = gym.make('CartPole-v0')
env.reset()
t = 1
while True:
    time.sleep(0.05)
    env.render()
    obs, r, done, inf = env.step(env.action_space.sample())
    if done:
        print("Resetting. Time:", t)
        env.reset()
        t = 1
    else:
        t += 1
