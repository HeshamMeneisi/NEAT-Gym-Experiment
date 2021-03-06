import os.path
import neat
import numpy as np
from neat_gym_exp import NEATGymExperiment
import roboschool
import matplotlib.pyplot as plt
from OpenGL import GLU


def interpret_a(a):
    return np.array(a)


def fitness(rec):
    return rec['reward'].mean()


# Load configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     os.path.join(os.path.dirname(__file__), 'config-pong'))

# Construct experiment
exp = NEATGymExperiment('RoboschoolPong-v1', config,
                        interpret_action=interpret_a,
                        runs_per_genome=1,
                        extract_fitness=fitness,
                        mode='parallel',
                        network=neat.nn.RecurrentNetwork
                        )

exp.exp_info(True)

winner = exp.run()

plt.plot(range(len(exp.f_record)), exp.f_record)
plt.ylabel('Fitness')
plt.xlabel('Generation')
plt.xlim(0, len(exp.f_record))
plt.xticks(range(len(exp.f_record)))
plt.show()

exp.test(winner)
