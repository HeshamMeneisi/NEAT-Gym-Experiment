import os.path
import neat
from neat_gym_exp import NEATGymExperiment


def int_a(a):
    return round(a[0])


def fitness(rec):
    return rec['reward'].mean()


# Load configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     os.path.join(os.path.dirname(__file__), 'config-pb'))

exp = NEATGymExperiment('CartPole-v0', config,
                        interpret_action=int_a,
                        runs_per_genome=100,
                        extract_fitness=fitness,
                        mode='parallel',
                        network=neat.nn.RecurrentNetwork)

winner = exp.run()

exp.test(winner)