import gym
import time
from gym import envs
import numpy as np
import roboschool
from OpenGL import GLU
import gym.envs.registration as reg
from mikobot import MiKoBot

reg.register("MiKo-v1", reward_threshold=2500, entry_point=MiKoBot, max_episode_steps=10000,
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

s1_thigh = 0
s2_thigh = 6
s3_thigh = 2
s4_thigh = 4

s1_leg = 1
s2_leg = 7
s3_leg = 3
s4_leg = 5

s1_int_thigh_angle = 90
s2_int_thigh_angle = 90
s3_int_thigh_angle = 90
s4_int_thigh_angle = 90

s1_int_leg_angle = 90
s2_int_leg_angle = 90
s3_int_leg_angle = 90
s4_int_leg_angle = 90

a = np.zeros(8)

def set_angle(angle, servo):
    value = angle / 90.0 - 1
    a[servo] = value

actions = [
    (s1_int_leg_angle - 30, s1_leg),
    (s1_int_thigh_angle, s1_thigh),
    (s1_int_leg_angle, s1_leg),

    (s2_int_leg_angle - 30, s2_leg),
    (s2_int_thigh_angle, s2_thigh),
    (s2_int_leg_angle, s2_leg),

    (s3_int_leg_angle - 30, s3_leg),
    (s3_int_thigh_angle, s3_thigh),
    (s3_int_leg_angle, s3_leg),

    (s4_int_leg_angle - 30, s4_leg),
    (s4_int_thigh_angle, s4_thigh),
    (s4_int_leg_angle, s4_leg),

    # change s2 and s4 for balance 
    (s2_int_leg_angle - 30, s2_leg),
    (s2_int_thigh_angle - 30, s2_thigh),
    (s2_int_leg_angle, s2_leg),
    (s4_int_leg_angle - 30, s4_leg),
    (s4_int_thigh_angle + 70, s4_thigh),
    (s4_int_leg_angle, s4_leg),

    # s2 back to initial position
    (s2_int_leg_angle - 30, s2_leg),
    (s2_int_thigh_angle, s2_thigh),
    (s2_int_leg_angle, s2_leg),

    # s4 and s2 will cause the bot to move forward from the rigth side
    (s4_int_leg_angle + 30, s4_leg),
    (s4_int_thigh_angle, s4_thigh),
    (s4_int_leg_angle, s4_leg),
    (s2_int_leg_angle + 30, s2_leg),
    (s2_int_thigh_angle + 20, s2_thigh),
    (s2_int_leg_angle, s2_leg),
    (s1_int_leg_angle + 30, s1_leg),
    (s1_int_thigh_angle + 80, s1_thigh),
    (s1_int_leg_angle, s1_leg),

    # change the position of s3 for balance
    (s3_int_leg_angle - 30, s3_leg),
    (s3_int_thigh_angle - 50, s3_thigh),
    (s3_int_leg_angle, s3_leg),

    # change the position of s1 for balance
    (s1_int_leg_angle - 30, s1_leg),
    (s1_int_thigh_angle - 20, s1_thigh),
    (s1_int_leg_angle, s1_leg),

    # bot moving forward from the left side
    (s4_int_leg_angle + 30, s4_leg),
    (s4_int_thigh_angle - 20, s4_thigh),
    (s4_int_leg_angle, s4_leg),
    (s2_int_leg_angle + 30, s2_leg),
    (s2_int_thigh_angle - 50, s2_thigh),
    (s2_int_leg_angle, s2_leg),
    (s1_int_leg_angle + 30, s1_leg),
    (s1_int_thigh_angle, s1_thigh),
    (s1_int_leg_angle, s1_leg),
    (s3_int_leg_angle + 30, s3_leg),
    (s3_int_thigh_angle, s3_thigh),
    (s3_int_leg_angle, s3_leg),
    (s2_int_leg_angle - 30, s2_leg),
    (s2_int_thigh_angle, s2_thigh),
    (s2_int_leg_angle, s2_leg),

    
]

env.render()
i = 0
while True:
    time.sleep(0.01)

    ca = actions[i % len(actions)]
    set_angle(ca[0], ca[1])

    env.render()
    obs, r, done, inf = env.step(a)
    if done:
        print("Resetting. Time:", t)
        env.reset()
        t = 1
        i = 0
    else:
        t += 1

    if t % 10 == 0:
        i += 1
