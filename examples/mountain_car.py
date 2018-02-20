import os.path
import neat
import numpy as np
from neat_gym_exp import NEATGymExperiment
import matplotlib.pyplot as plt


def interpret_a(a):
    return np.argmax(a)


def fitness(rec):
    # Agents who succeed at least once get 0.5 fitness, those who succeed in all runs get 0.5 more fitness
    m_dist = [np.max(np.array(run)[:, 0]) for run in rec['obs']]
    return np.mean(m_dist) + np.max(m_dist)


# Load configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     os.path.join(os.path.dirname(__file__), 'config-mc'))

# Construct experiment
exp = NEATGymExperiment('MountainCar-v0', config,
                        interpret_action=interpret_a,
                        runs_per_genome=5,
                        extract_fitness=fitness,
                        mode='parallel'
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
