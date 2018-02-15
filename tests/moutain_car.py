import os.path
import neat
import numpy as np
from neat_gym_exp import NEATGymExperiment
import matplotlib.pyplot as plt

# Load configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     os.path.join(os.path.dirname(__file__), 'config-mc'))

exp = NEATGymExperiment('MountainCar-v0', config, lambda a: np.argmax(a),
                        5, lambda rec: rec['reward'].mean())

winner = exp.run()

plt.scatter(range(len(exp.f_record)), exp.f_record)
plt.ylabel('Fitness')
plt.xlabel('Generation')
plt.xlim(0, len(exp.f_record))
plt.xticks(range(len(exp.f_record)))
plt.show()

exp.test(winner)