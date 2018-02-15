import os.path
import neat
from neat_gym_exp import NEATGymExperiment

# Load configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     os.path.join(os.path.dirname(__file__), 'config-pb'))

exp = NEATGymExperiment('CartPole-v0', config, lambda a: round(a[0]), 100, lambda rec: rec['reward'].mean())

winner = exp.run()

exp.test(winner)
