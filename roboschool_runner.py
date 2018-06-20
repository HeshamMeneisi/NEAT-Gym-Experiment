import neat
import numpy as np
from pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.es_hyperneat.es_hyperneat import ESNetwork
import pickle
import os


def eval_genomes(all, config):
    for key, g in all.items():
        eval_fitness(g, config)


def eval_fitness(g, config):
    cppn = neat.nn.FeedForwardNetwork.create(g, config)
    network = ESNetwork(substrate, cppn, params)
    net = network.create_phenotype_network()

    fitnesses = []

    for i in range(trials):
        ob = env.reset()
        net.reset()

        total_reward = 0

        for j in range(max_steps):
            for k in range(network.activations):
                o = net.activate(ob)
            action = np.array(o)
            ob, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        fitnesses.append(total_reward)

    g.fitness = np.array(fitnesses).mean()

    return g.fitness


# Initialize population attaching statistics reporter.
def ini_pop(state, stats, config, output):
    pop = neat.population.Population(config, state)
    if output:
        pop.add_reporter(neat.reporting.StdOutReporter(True))
    pop.add_reporter(stats)
    return pop


# Generic OpenAI Gym runner for ES-HyperNEAT.
def run_es(gens, _env, _max_steps, config, _params, _substrate, _max_trials=100, output=True, mode='parallel'):
    global env, substrate, params, max_trails, trials, max_steps

    env = _env
    substrate = _substrate
    params = _params
    max_trials = _max_trials
    max_steps = _max_steps

    trials = 1

    # Create population and train the network. Return winner of network running 100 episodes.
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_pop(None, stats_one, config, output)

    stats_ten = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_ten, config, output)
    trials = 10

    # eval_genomes(pop.population, config)

    p_th = neat.ThreadedEvaluator(8, eval_fitness)
    p_par = neat.ParallelEvaluator(8, eval_fitness)

    gen = 0

    while not pop.best_genome or pop.best_genome.fitness < pop.config.fitness_threshold:
        if mode == 'threaded':
            print("Mode: Threaded")

            winner_ten = pop.run(p_th.evaluate, 1)
        elif mode == 'parallel':
            print("Mode: Parallel")

            winner_ten = pop.run(p_par.evaluate, 1)
        else:
            print("Mode: Default")
            winner_ten = pop.run(eval_genomes, 1)

        if gen % 10 == 0:
            if not os.path.exists('./logs'):
                os.mkdir('./logs')
            with open('./logs/' + str(gen), 'wb') as f:
                pickle.dump(pop, f)

        gen += 1

    if max_trials is 0:
        return winner_ten, (stats_one, stats_ten)

    stats_hundred = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_hundred, config, output)
    trials = max_trials
    winner_hundred = pop.run(eval_fitness, gens)
    return winner_hundred, (stats_one, stats_ten, stats_hundred)


# Generic OpenAI Gym runner for HyperNEAT.
def run_hyper(gens, env, max_steps, config, substrate, activations, max_trials=100, activation="sigmoid", output=True,
              mode='parallel'):
    trials = 1

    def eval_fitness(genomes, config):

        for idx, g in genomes:
            cppn = neat.nn.FeedForwardNetwork.create(g, config)
            net = create_phenotype_network(cppn, substrate, activation)

            fitnesses = []

            for i in range(trials):
                ob = env.reset()
                net.reset()

                total_reward = 0

                for j in range(max_steps):
                    for k in range(activations):
                        o = net.activate(ob)
                    action = np.argmax(o)
                    ob, reward, done, info = env.step(action)
                    total_reward += reward
                    if done:
                        break
                fitnesses.append(total_reward)

            g.fitness = np.array(fitnesses).mean()

    # Create population and train the network. Return winner of network running 100 episodes.
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_pop(None, stats_one, config, output)
    pop.run(eval_fitness, gens)

    stats_ten = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_ten, config, output)
    trials = 10

    p_th = neat.ThreadedEvaluator(8, eval_fitness)
    p_par = neat.ParallelEvaluator(8, eval_fitness)

    gen = 0

    while not pop.best_genome or pop.best_genome.fitness < pop.config.fitness_threshold:
        if mode == 'threaded':
            print("Mode: Threaded")

            winner_ten = pop.run(p_th.evaluate, 1)
        elif mode == 'parallel':
            print("Mode: Parallel")

            winner_ten = pop.run(p_par.evaluate, 1)
        else:
            print("Mode: Default")
            winner_ten = pop.run(eval_genomes, 1)

        if gen % 10 == 0:
            if not os.path.exists('./logs'):
                os.mkdir('./logs')
            with open('./logs/' + str(gen), 'wb') as f:
                pickle.dump(pop, f)

        gen += 1

    if max_trials is 0:
        return winner_ten, (stats_one, stats_ten)


    stats_hundred = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_hundred, config, output)
    trials = max_trials
    winner_hundred = pop.run(eval_fitness, gens)
    return winner_hundred, (stats_one, stats_ten, stats_hundred)


# Generic OpenAI Gym runner for NEAT.
def run_neat(gens, env, max_steps, config, max_trials=100, output=True, mode='parallel'):
    trials = 1

    def eval_fitness(g, config):
        net = neat.nn.FeedForwardNetwork.create(g, config)

        fitnesses = []

        for i in range(trials):
            ob = env.reset()

            total_reward = 0

            for j in range(max_steps):
                o = net.activate(ob)
                action = np.argmax(o)
                ob, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break
            fitnesses.append(total_reward)

        g.fitness = np.array(fitnesses).mean()

    # Create population and train the network. Return winner of network running 100 episodes.
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_pop(None, stats_one, config, output)
    pop.run(eval_fitness, gens)

    stats_ten = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_ten, config, output)
    trials = 10

    p_th = neat.ThreadedEvaluator(8, eval_fitness)
    p_par = neat.ParallelEvaluator(8, eval_fitness)

    gen = 0

    while not pop.best_genome or pop.best_genome.fitness < pop.config.fitness_threshold:
        if mode == 'threaded':
            print("Mode: Threaded")

            winner_ten = pop.run(p_th.evaluate, 1)
        elif mode == 'parallel':
            print("Mode: Parallel")

            winner_ten = pop.run(p_par.evaluate, 1)
        else:
            print("Mode: Default")
            winner_ten = pop.run(eval_genomes, 1)

        if gen % 10 == 0:
            if not os.path.exists('./logs'):
                os.mkdir('./logs')
            with open('./logs/' + str(gen), 'wb') as f:
                pickle.dump(pop, f)

        gen += 1


    if max_trials is 0:
        return winner_ten, (stats_one, stats_ten)

    stats_hundred = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_hundred, config, output)
    trials = max_trials
    winner_hundred = pop.run(eval_fitness, gens)
    return winner_hundred, (stats_one, stats_ten, stats_hundred)

