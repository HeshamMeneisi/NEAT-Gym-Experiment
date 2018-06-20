import neat
import logging
import _pickle as pickle
import gym
import time
import numpy as np
from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate
from roboschool_runner import run_es
from pureples.es_hyperneat.es_hyperneat import ESNetwork
import gym.envs.registration as reg
from mikobot import MiKoBot
from OpenGL import GLU

reg.register("MiKo-v1", reward_threshold=2500, entry_point=MiKoBot, max_episode_steps=1000,
             tags={"pg_complexity": 8000000})

# Network input and output coordinates.
input_coordinates = [(8, 2), (10, 0), (8, 8), (10, 10), (2, 8), (0, 10), (2, 2), (0, 0), (3, 0), (7, 0)]
output_coordinates = [(8, 2), (10, 0), (8, 8), (10, 10), (2, 8), (0, 10), (2, 2), (0, 0)]

individual = 460

sub = Substrate(input_coordinates, output_coordinates)

# ES-HyperNEAT specific parameters.
params = {"initial_depth": 0,
          "max_depth": 1,
          "variance_threshold": 0.03,
          "band_threshold": 0.3,
          "iteration_level": 1,
          "division_threshold": 0.5,
          "max_weight": 8.0,
          "activation": "sigmoid"}

# Config for CPPN.
config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_cppn_miko')


# If run as script.
if __name__ == '__main__':
    # Setup logger and environment.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    env = gym.make("MiKo-v1")

    with open('./logs/' + str(individual), 'rb') as f:
        winner = pickle.load(f).best_genome

    # Save CPPN if wished reused and draw it + winner to file.
    cppn = neat.nn.FeedForwardNetwork.create(winner, config)
    network = ESNetwork(sub, cppn, params)
    net = network.create_phenotype_network(filename="es_hyperneat_miko_small_winner")
    draw_net(cppn, filename="es_hyperneat_miko_small_cppn")
    with open('es_hyperneat_miko_small_cppn.pkl', 'wb') as output:
        pickle.dump(cppn, output)

    env.reset()

    obs, r, done, inf = env.step(env.action_space.sample())
    env.render()
    t = 0
    while True:
        time.sleep(0.05)
        a = net.activate(obs)
        obs, r, done, inf = env.step(np.array(a))
        env.render()
        if done:
            print("Resetting. Time:", t)
            env.reset()
            t = 1
        else:
            t += 1