# It's a script that's not necessary to train and evaluate the PCN, but it's useful to perform grid-search on hyper-parameters.
import argparse
from itertools import product
from pathlib import Path
import random

import numpy as np
from train_pcn import main as pcn_main
from train_lcn import main as lcn_main
import time
import torch

### Hyperparameters for Dilemma
# batch_sizes = [128, 256]
# lrs = [1e-1]
# er_episodes = [25]
# max_buffer_sizes = [50]
# model_updates = [10]
# timesteps = [1000]
###

# ### Hyperparameters for Amsterdam
batch_sizes = [128, 256]
lrs = [1e-1, 1e-2]
er_episodes = [50, 100]
max_buffer_sizes = [50, 100]
# model_updates = [5, 10]
model_updates = [5]
# nr_layers = [1, 2, 3]
nr_layers = [1, 2]
hidden_dims = [64, 128]
timesteps = [30000]
train_mode = 'uniform'
num_explore_episodes = None
distance_ref = 'interpolate'

settings = [batch_sizes, lrs, er_episodes, max_buffer_sizes, model_updates, nr_layers, hidden_dims, timesteps]

# All combinations of hyperparameters
combinations = list(product(*settings))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"MO-TNDP Hyperparameter Search")
    # Acceptable values: 'dilemma', 'amsterdam'
    parser.add_argument('--env', default='dilemma', type=str)
    # For amsterdam environment we have different groups files (different nr of objectives)
    parser.add_argument('--nr_groups', default=5, type=int)
    # Starting location of the agent
    parser.add_argument('--starting_loc_x', default=None, type=int)
    parser.add_argument('--starting_loc_y', default=None, type=int)
    # Episode horizon -- used as a proxy of both the budget and the number of stations (stations are not really costed)
    # parser.add_argument('--nr_stations', default=9, type=int)
    parser.add_argument('--model', default=None, type=str) # PCN or LCN
    parser.add_argument('--num_policies', default=10, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--no_log', action='store_true', default=False)
    parser.add_argument('--random', action='store_true', default=False, help='if true, the hyperparameter search will be random instead of exhaustive')
    parser.add_argument('--lcn_lambda', default=None, type=float)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Some values are hardcoded for each environment (this is flexible, but we don't want to have to pass 100 arguments to the script)
    if args.env == 'dilemma':
        args.city_path = Path(f"./envs/mo-tndp/cities/dilemma_5x5")
        args.nr_stations = 9
        args.gym_env = 'motndp_dilemma-v0'
        args.groups_file = "groups.txt"
        args.num_step_episodes = 10
        args.ignore_existing_lines = True
        args.experiment_name = f"{args.model}-Dilemma"
        args.scaling_factor = np.array([1, 1, 0.1])
        args.ref_point = np.array([0, 0])
        args.max_return=np.array([1, 1])
        args.pf_plot_limits = [0, 0.5]
        args.update_interval = None
        args.cd_threshold = 0.2
        args.lcn_lambda = args.lcn_lambda
    elif args.env == 'amsterdam':
        args.city_path = Path(f"./envs/mo-tndp/cities/amsterdam")
        args.nr_stations = 20
        args.gym_env = 'motndp_amsterdam-v0'
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.num_step_episodes = 10
        args.ignore_existing_lines = True
        args.experiment_name = f"{args.model}-Amsterdam"
        args.scaling_factor = np.array([100] * args.nr_groups + [0.01])
        args.ref_point = np.array([0] * args.nr_groups)
        args.max_return=np.array([1] * args.nr_groups)
        # args.pf_plot_limits = [0, 0.015]
        args.pf_plot_limits = None
        args.update_interval = None
        args.cd_threshold = 0.2
        args.lcn_lambda = args.lcn_lambda
    elif args.env == 'xian':
        args.city_path = Path(f"./envs/mo-tndp/cities/xian")
        args.nr_stations = 20
        args.gym_env = 'motndp_xian-v0'
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.num_step_episodes = 10
        args.ignore_existing_lines = True
        args.experiment_name = f"{args.model}-Xian"
        args.scaling_factor = np.array([100] * args.nr_groups + [0.01])
        args.ref_point = np.array([0] * args.nr_groups)
        args.max_return=np.array([1] * args.nr_groups)
        args.pf_plot_limits = None
        args.update_interval = None
        args.cd_threshold = 0.2
        args.distance_ref = distance_ref
        args.lcn_lambda = args.lcn_lambda
    
    if args.starting_loc_x is not None and args.starting_loc_y is not None:
        args.starting_loc = (args.starting_loc_x, args.starting_loc_y)
    else:
        args.starting_loc = None

    counter = 0
    running_times = []
    total_runs = len(combinations)
    if args.random:
        random.shuffle(combinations)
    for batch_size, lr, er_ep, max_buffer_size, model_update, nr_l, hidden_d, timestep in combinations:
        counter += 1
        start_time = time.time()
        # Average running time of the last 5 runs
        avg_runtime = np.mean(running_times[-5:])
        print(f'Run {counter}/{total_runs} | Avg running time: {avg_runtime} | Estimated Time left: {(total_runs - counter)*avg_runtime / 60} minutes | env: {args.env} batch_size: {batch_size}, lr: {lr}, er_episodes: {er_ep}, max_buffer_size: {max_buffer_size}, model_update: {model_update}, hidden_size: {hidden_d}, timestep: {timestep}')
        args.batch_size = batch_size
        args.lr = lr
        args.num_er_episodes = er_ep
        args.max_buffer_size = max_buffer_size
        args.num_model_updates = model_update
        args.timesteps = timestep
        args.hidden_dim = hidden_d
        args.nr_layers = nr_l
        args.train_mode = train_mode
        args.num_explore_episodes = num_explore_episodes

        if args.model == 'PCN':
            pcn_main(args)
        elif args.model == 'LCN':
            lcn_main(args)

        execution_time = time.time() - start_time
        running_times.append(execution_time)
