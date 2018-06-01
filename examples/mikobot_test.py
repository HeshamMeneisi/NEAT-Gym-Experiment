import os.path
import neat
import numpy as np
from neat_gym_exp import NEATGymExperiment
from OpenGL import GLU

import gym.envs.registration as reg
from mikobot import MiKoBot

reg.register("MiKo-v1", reward_threshold=2500, entry_point=MiKoBot, max_episode_steps=1000,
             tags={"pg_complexity": 8000000})


def int_a(a):
    return np.array(a)


# Load configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     os.path.join(os.path.dirname(__file__), 'config-miko'))

# Construct experiment
exp = NEATGymExperiment('MiKo-v1', config,
                        interpret_action=int_a,
                        extract_fitness=lambda x: x,
                        network=neat.nn.MLRecurrentNetwork
                        )

exp.test(exp.p.best_genome)