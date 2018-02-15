import neat
import pickle
import gym
import numpy as np
import time


class NEATGymExperiment:

    def __init__(self, gym_experiment, neat_config, interpret_action, runs_per_genom, extract_fittness, render=False):
        self.exp_name = gym_experiment
        self.env = gym.make(gym_experiment)
        self.config = neat_config
        self.interpret_action = interpret_action
        self.rpg = runs_per_genom
        self.extract_f = extract_fittness
        self.render = render

        self.record = dict()
        self.record['reward'] = np.zeros(runs_per_genom)

        # Create the population, which is the top-level object for a NEAT run.
        self.p = neat.Population(neat_config)

        # Add a stdout reporter to show progress in the terminal.
        self.p.add_reporter(neat.StdOutReporter(False))

        self.f_record = []

    def get_action(self, genome, obs):
        net = neat.nn.FeedForwardNetwork.create(genome, self.config)
        return self.interpret_action(net.activate(obs))

    def get_fitness(self, genome):
        for i in range(self.rpg):
            done = False
            obs = self.env.reset()
            self.record['reward'][i] = 0
            while not done:
                if self.render:
                    self.env.render()
                a = self.get_action(genome, obs)
                obs, r, done, inf = self.env.step(a)
                self.record['reward'][i] += r
        return self.extract_f(self.record)

    def eval_genomes(self, genomes, config):
        self.config = config

        for genome_id, genome in genomes:
            genome.fitness = self.get_fitness(genome)

        if self.p.best_genome:
            self.f_record.append(self.p.best_genome.fitness)

    def run(self):
        # Run until a solution is found.
        winner = self.p.run(self.eval_genomes)

        self.f_record.append(winner.fitness)

        pickle.dump(winner, open('./' + self.exp_name + '_winner.dat', 'wb'))

        pickle.dump(self.f_record, open('./' + self.exp_name + '_flog.dat', 'wb'))

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        return winner

    def test(self, genome):
        obs = self.env.reset()
        t = 1
        while True:
            time.sleep(0.05)
            self.env.render()
            obs, r, done, inf = self.env.step(self.get_action(genome, obs))
            if done:
                print("Resetting. Steps:", t)
                obs = self.env.reset()
                t = 1
            else:
                t += 1
