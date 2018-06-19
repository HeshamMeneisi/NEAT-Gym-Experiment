import gym
import time
from gym import envs
import numpy as np
import roboschool
from OpenGL import GLU
import gym.envs.registration as reg
from mikobot import MiKoBot

reg.register("MiKo-v1", reward_threshold=2500, entry_point=MiKoBot, max_episode_steps=1000,
             tags={"pg_complexity": 8000000, 'no_death': True})

print([e.id for e in envs.registry.all()])

env = gym.make('MiKo-v1')
info = {
            'obs_space': env.observation_space,
            'obs_s_max': env.observation_space.high if hasattr(env.observation_space, 'high') else None,
            'obs_s_min': env.observation_space.low if hasattr(env.observation_space, 'low') else None,
            'act_space': env.action_space,
            'act_s_max': env.action_space.high if hasattr(env.action_space, 'high') else None,
            'act_s_min': env.action_space.low if hasattr(env.action_space, 'high') else None,
            'rw_range': env.reward_range,
            'meta': env.metadata
        }
print(info)
env.reset()

t = 1
up = np.array([0, -1, 0, -1, 0, -1, 0, -1])
down = np.array([0, 1, 0, 1, 0, 1, 0, 1])
whirl = np.array([1, 0, 1, 0, 1, 0, 1, 0])
hover = np.array([0, 0, 0, 0, 0, 0, 0, 0])
crouch = np.array([0, 1, 0, 1, 0, 1, 0, 1])
left_up = np.array([0, -1, 0, -1, 0, 1, 0, 1])
right_up = np.array([0, 1, 0, 1, 0, -1, 0, -1])
front_up = np.array([0, -1, 0, 1, 0, 1, 0, -1])
back_up = np.array([0, 1, 0, -1, 0, -1, 0, 1])
left_front_raised = np .array([0, 1, 0, -1, 0, -1, 0, -1])

x = 25

env.render()
while True:
    time.sleep(0.01)
    #a = env.action_space.sample()

    if t < x:
        a = left_up
    elif t < 2*x:
        a = front_up
    elif t < 3*x:
        a = right_up
    elif t < 4*x:
        a = back_up
    elif t < 5*x:
        a = hover
    elif t < 6*x:
        a = up
    elif t < 7*x:
        a = hover
    elif t < 8*x:
        a = down
    elif t < 9*x:
        a = whirl
    elif t < 10*x:
        a = -whirl
    elif t < 11*x:
        a = crouch
    elif t > 14*x:
        a = up

    env.render()
    obs, r, done, inf = env.step(a)
    if done:
        print("Resetting. Time:", t)
        env.reset()
        t = 1
    else:
        t += 1
