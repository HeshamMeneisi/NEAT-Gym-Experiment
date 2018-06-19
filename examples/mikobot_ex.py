import os.path
import neat
import numpy as np
from neat_gym_exp import NEATGymExperiment
import roboschool
import matplotlib.pyplot as plt
from OpenGL import GLU
import gym.envs.registration as reg
from mikobot import MiKoBot

reg.register("MiKo-v1", reward_threshold=2500, entry_point=MiKoBot, max_episode_steps=1000,
             tags={"pg_complexity": 8000000})


def int_a(a):
    return np.array(a)


def fitness(rec):
    f = rec['reward'].sum(axis=1).mean()
    return f


# Load configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     os.path.join(os.path.dirname(__file__), 'config-miko'))

# Construct experiment
exp = NEATGymExperiment('MiKo-v1', config,
                        interpret_action=int_a,
                        runs_per_genome=2,
                        extract_fitness=fitness,
                        mode='parallel',
                        instances=7,
                        # render_all=True,
                        # network=neat.nn.GRUNetwork,
                        # starting_gen=0
                        )


exp.exp_info(True)

winner = exp.run()

plt.plot(range(len(exp.f_record)), exp.f_record)
plt.ylabel('Fitness')
plt.xlabel('Generation')
plt.xlim(0, len(exp.f_record))
plt.xticks(range(0, len(exp.f_record), len(exp.f_record) // 10))
plt.show()

exp.test(winner)
