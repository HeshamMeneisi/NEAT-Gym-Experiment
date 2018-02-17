import os.path
import neat
from neat_gym_exp import NEATGymExperiment

# Load configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     os.path.join(os.path.dirname(__file__), 'config-pb'))

exp = NEATGymExperiment('CartPole-v0', config,
                        interpret_action=lambda a: round(a[0]),
                        runs_per_genome=100,
                        extract_fitness=lambda rec: rec['reward'].mean())

winner = exp.run()

exp.test(winner)
