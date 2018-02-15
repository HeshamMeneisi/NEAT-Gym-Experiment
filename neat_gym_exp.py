import neat
import pickle
import gym
import numpy as np
import time


class NEATGymExperiment:

    def __init__(self, gym_experiment, neat_config, interpret_action, runs_per_genome, extract_fitness, render=False,
                 render_delay=0.005, verbose=False, render_champ=False):

        assert hasattr(interpret_action, '__call__')
        assert runs_per_genome > 0
        assert hasattr(extract_fitness, '__call__')

        self.exp_name = gym_experiment
        self.env = gym.make(gym_experiment)
        self.config = neat_config
        self.interpret_action = interpret_action
        self.rpg = runs_per_genome
        self.extract_f = extract_fitness
        self.render = render
        self.r_delay = render_delay
        self.verbose = verbose
        self.render_champ = render_champ

        self.record = dict()
        self.record['reward'] = np.zeros(runs_per_genome)
        self.record['obs'] = []

        # Create the population, which is the top-level object for a NEAT run.
        self.p = neat.Population(neat_config)

        # Add a stdout reporter to show progress in the terminal.
        self.p.add_reporter(neat.StdOutReporter(False))

        self.f_record = []

    def get_action(self, genome, obs):
        """
        Get the agent's action using the interpret_action function supplied to the initializer.
        :param genome: The agent's genome
        :param obs: The observation from our environment
        :return: Action
        """
        net = neat.nn.FeedForwardNetwork.create(genome, self.config)
        return self.interpret_action(net.activate(obs))

    def get_fitness(self, genome):
        """
        Calculate the fitness of the given genome using the extract_fitness function supplied to the initializer.
        :param genome: The agent's genome
        :return: Fitness
        """
        self.record['obs'] = []
        for i in range(self.rpg):
            done = False
            obs = self.env.reset()
            self.record['reward'][i] = 0
            self.record['obs'].append([])
            self.record['obs'][i].append(obs.tolist())
            while not done:
                if self.render:
                    self.env.render()
                a = self.get_action(genome, obs)
                obs, r, done, inf = self.env.step(a)
                self.record['reward'][i] += r
                self.record['obs'][i].append(obs.tolist())

        return self.extract_f(self.record)

    def eval_genomes(self, genomes, config):
        """
        This function is called by neat-python's Population.run()
        :param genomes: A list of genomes
        :param config: The NEAT configuration
        :return:
        """
        self.config = config

        for genome_id, genome in genomes:
            genome.fitness = self.get_fitness(genome)
            if self.verbose:
                print(genome_id, genome.fitness)

        if self.p.best_genome:
            self.f_record.append(self.p.best_genome.fitness)
            if self.render_champ:
                self.test(self.p.best_genome, n=1)

    def run(self):
        """
        Run until the fitness threshold is reached.
        :return:
        """
        winner = self.p.run(self.eval_genomes)

        self.f_record.append(winner.fitness)

        pickle.dump(winner, open('./' + self.exp_name + '_winner.dat', 'wb'))

        pickle.dump(self.f_record, open('./' + self.exp_name + '_flog.dat', 'wb'))

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        return winner

    def test(self, genome, n=None):
        """
        Run the environment using the given agent n times.
        :param genome: The agent's genome
        :param n: Number of test runs. None means infinite.
        :return:
        """
        obs = self.env.reset()
        k = 0
        t = 1
        while True:
            time.sleep(self.r_delay)
            self.env.render()
            obs, r, done, inf = self.env.step(self.get_action(genome, obs))
            if done:
                print("Resetting. Steps:", t)
                k += 1

                if n and k >= n:
                    break

                obs = self.env.reset()
                t = 1
            else:
                t += 1
