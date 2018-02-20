import neat
import pickle
import gym
import numpy as np
import time
from copy import deepcopy
from threading import Lock

class NEATGymExperiment:

    def __init__(self, gym_experiment, neat_config, extract_fitness, runs_per_genome=1, interpret_action=None,
                 multiplayer=False, server_guide=None, n_players=1, verbose=False, render_champ=False, render_all=False,
                 render_delay=0.005, render_max_frames=200, mode='default', instances=4):

        assert interpret_action is None or hasattr(interpret_action, '__call__')
        assert runs_per_genome > 0
        assert hasattr(extract_fitness, '__call__')
        assert mode in ['default', 'threaded', 'parallel']

        self.exp_name = gym_experiment
        self.env = gym.make(gym_experiment)
        self.config = neat_config
        self.interpret_action = interpret_action
        self.rpg = runs_per_genome
        self.extract_f = extract_fitness
        self.multiplayer = multiplayer
        self.server_guide = server_guide
        self.n_players = n_players
        self.verbose = verbose
        self.render = render_all
        self.render_champ = render_champ
        self.r_delay = render_delay
        self.render_max_frames = render_max_frames
        self.mode = mode
        self.instances = instances

        if mode == 'threaded':
            self.pool_lock = Lock()
            self.pool_avail = [True] * instances
            self.pool_env = []
            for i in range(instances):
                self.pool_env.append(deepcopy(self.env))

        self.record = dict()
        self.record['reward'] = np.zeros(runs_per_genome)
        s = list(self.env.observation_space.shape)
        s.insert(0, self.env._max_episode_steps)
        s.insert(0, runs_per_genome)
        self.record['obs'] = np.zeros(tuple(s))

        if multiplayer:
            self.env.unwrapped.multiplayer(self.env, game_server_guid=server_guide, player_n=n_players)

        # Create the population, which is the top-level object for a NEAT run.
        self.p = neat.Population(neat_config)

        # Add a stdout reporter to show progress in the terminal.
        self.p.add_reporter(neat.StdOutReporter(False))

        self.f_record = []

    def acquire_env(self):
        self.pool_lock.acquire()
        for i in range(self.instances):
            if self.pool_avail[i]:
                self.pool_avail[i] = False
                self.pool_lock.release()
                return i, self.pool_env[i]

    def release_env(self, id):
        self.pool_avail[id] = True

    def get_action(self, genome, obs):
        """
        Get the agent's action using the interpret_action function supplied to the initializer.
        :param genome: The agent's genome
        :param obs: The observation from our environment
        :return: Action
        """
        if self.config.genome_config.feed_forward:
            net = neat.nn.FeedForwardNetwork.create(genome, self.config)
        else:
            net = neat.nn.RecurrentNetwork.create(genome, self.config)

        if self.interpret_action is not None:
            return self.interpret_action(net.activate(obs))

        return net.activate(obs)

    def get_fitness(self, genome):
        """
        Calculate the fitness of the given genome using the extract_fitness function supplied to the initializer.
        :param genome: The agent's genome
        :return: Fitness
        """
        if self.mode == 'threaded':
            env_id, env = self.acquire_env()
            record = deepcopy(self.record)
        else:
            env = self.env
            record = self.record

        for i in range(self.rpg):
            done = False
            obs = env.reset()
            record['reward'][i] = 0
            record['obs'][i, 0] = obs
            t = 1
            while not done:
                if self.render and t < self.render_max_frames:
                    env.render()
                a = self.get_action(genome, obs)
                obs, r, done, inf = env.step(a)
                record['reward'][i] += r
                record['obs'][i, t] = obs

        if self.mode == 'threaded':
            self.release_env(env_id)

        return self.extract_f(record)

    def eval_genome(self, genome, config=None):
        """
        This function evaluates a single genome. Can be called by ThreadedEvaluator and ParallelEvaluator
        :param genome_id:
        :param genome:
        :return:
        """
        fitness = self.get_fitness(genome)

        if self.verbose:
            print(genome.id, fitness)

        return fitness

    def eval_genomes(self, genomes, config):
        """
        This function is called by neat-python's Population.run()
        :param genomes: A list of genomes
        :param config: The NEAT configuration
        :return:
        """
        self.config = config

        for genome_id, genome in genomes:
            genome.fitness = self.eval_genome(genome)

        if self.p.best_genome:
            self.f_record.append(self.p.best_genome.fitness)
            if self.render_champ:
                print("Testing champion.")
                self.test(self.p.best_genome, n=1)

    def run(self):
        """
        Run until the fitness threshold is reached.
        :return:
        """
        if self.mode == 'threaded':
            print("Mode: Threaded")
            pe = neat.ThreadedEvaluator(self.instances, self.eval_genome)
            winner = self.p.run(pe.evaluate)
        elif self.mode == 'parallel':
            print("Mode: Parallel")
            pe = neat.ParallelEvaluator(self.instances, self.eval_genome)
            winner = self.p.run(pe.evaluate)
        else:
            print("Mode: Default")
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
            if done or t >= self.render_max_frames:
                print("Resetting. Steps:", t)
                k += 1

                if n and k >= n:
                    break

                obs = self.env.reset()
                t = 1
            else:
                t += 1
        self.env.close()

    def exp_info(self, _print=False):
        """
        Return information about the loaded environment.
        :return:
        """
        info = {
            'obs_space': self.env.observation_space,
            'obs_s_max': self.env.observation_space.high if hasattr(self.env.observation_space, 'high') else None,
            'obs_s_min': self.env.observation_space.low if hasattr(self.env.observation_space, 'low') else None,
            'act_space': self.env.action_space,
            'act_s_max': self.env.action_space.high if hasattr(self.env.action_space, 'high') else None,
            'act_s_min': self.env.action_space.low if hasattr(self.env.action_space, 'high') else None,
            'rw_range': self.env.reward_range,
            'meta': self.env.metadata
        }

        if _print:
            for k, v in info.items():
                print(k, ':', v)

        return info
