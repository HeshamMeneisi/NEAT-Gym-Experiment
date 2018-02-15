# Overview
This is a simple implementation of a Gym experiment using neat-python

# Dependencies

python

gym

neat-python

# Running

You can run any of the tests included directly 
```bash
python exp_name
```

# Test Description

`test_neat.py`

Running this file will test the python-neat package using an XOR problem.

`test_gym.py`

Running this file will test the Gym package using the Pole-Balancing environment.

`test_gym_neat.py`

This file will test the `NEATGymExperiment` implementation using the Pole-Balancing experiment.

`mountain_car.py`

This will run an experiment using the MountainCar-v0 environment. It will stop as soon as an agent is able to reach the target in any time.

# Example

Running `mountain_car.py` untill convergence will first display a fitness over generations plot like this one:

![alt text](https://raw.githubusercontent.com/HeshamMeneisi/NEAT-Gym-Experiment/master/fplot.png)

Then the winning agent will take control of the environment until you terminate:

![alt text](https://raw.githubusercontent.com/HeshamMeneisi/NEAT-Gym-Experiment/master/mc.gif)

# NEATGymExperiment

**Constructor**

|  Parameter  | Default | Description  |
|---|---|--|
| `gym_experiment`  |  | The id of the gym experiment.  |
| `neat_config` |  | The neat-python configuration.  |
| `interpret_action` | |  A callable that infers action from network output. |
| `runs_per_genome` | | Experiments run for every genome. |
| `extract_fitness` | | A callable that extracts fitness from the experiment record.
| `render=False` | False | Whether to render experiments.
| `render_delay` | 0.005 | The delay per frame when rendering.
| `verbose` | False | Print extra information.
| `render_champ` | False | Render the champion of every generation.
 
 To run the experiment:
 
`winner = exp.run()`

The winning champion is always saved in a file with the name `experiment_id_winner.dat` and the best fitness for each generation is saved in `experiment_id_flog.dat`