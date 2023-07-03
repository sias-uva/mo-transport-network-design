# It's a script that's not necessary to train and evaluate the PCN, but it's useful to perform grid-search on hyper-parameters.
import argparse
from pathlib import Path
import random

import numpy as np
from train_pcn import main
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
batch_sizes = [256, 512, 1024]
lrs = [1e-1, 1e-2]
er_episodes = [50, 100]
max_buffer_sizes = [50, 100]
model_updates = [5, 10]
nr_layers = [1, 2]
hidden_dims = [64, 128]
timesteps = [30000]
train_mode = 'disttofront'
# ###

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MO PCN - TNDP Hyperparameter Search")
    # Acceptable values: 'dilemma', 'amsterdam'
    parser.add_argument('--env', default='dilemma', type=str)
    # For amsterdam environment we have different groups files (different nr of objectives)
    parser.add_argument('--nr_groups', default=5, type=int)
    # Starting location of the agent
    parser.add_argument('--starting_loc_x', default=None, type=int)
    parser.add_argument('--starting_loc_y', default=None, type=int)
    # Episode horizon -- used as a proxy of both the budget and the number of stations (stations are not really costed)
    # parser.add_argument('--nr_stations', default=9, type=int)
    parser.add_argument('--num_policies', default=10, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--no_log', action='store_true', default=False)

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
        args.experiment_name = "PCN-Dilemma"
        args.scaling_factor = np.array([1, 1, 0.1])
        args.ref_point = np.array([0, 0])
        args.max_return=np.array([1, 1])
        args.pf_plot_limits = [0, 0.5]
    elif args.env == 'amsterdam':
        args.city_path = Path(f"./envs/mo-tndp/cities/amsterdam")
        args.nr_stations = 20
        args.gym_env = 'motndp_amsterdam-v0'
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.num_step_episodes = 10
        args.ignore_existing_lines = True
        args.experiment_name = "PCN-Amsterdam"
        args.scaling_factor = np.array([1] * args.nr_groups + [0.01])
        args.ref_point = np.array([0] * args.nr_groups)
        args.max_return=np.array([1] * args.nr_groups)
        args.pf_plot_limits = [0, 0.5]
    
    if args.starting_loc_x is not None and args.starting_loc_y is not None:
        args.starting_loc = (args.starting_loc_x, args.starting_loc_y)
    else:
        args.starting_loc = None

    total_runs = len(batch_sizes) * len(lrs) * len(er_episodes) * len(max_buffer_sizes) * len(model_updates) * len(hidden_dims) * len(timesteps)
    counter = 0
    running_times = []
    for batch_size in batch_sizes:
        for lr in lrs:
            for er_ep in er_episodes:
                for max_buffer_size in max_buffer_sizes:
                    for model_update in model_updates:
                        for hidden_d in hidden_dims:
                            for nr_l in nr_layers:
                                for timestep in timesteps:
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

                                    main(args)

                                    execution_time = time.time() - start_time
                                    running_times.append(execution_time)

#%% OLD CODE --  Dilemma Hypeparameter Search
# batch_sizes = [128, 256, 512]
# lrs = [1e-1, 1e-2, 1e-3, 1e-4]
# er_episodes = [25, 50, 100]

# for batch_size in batch_sizes:
#     for lr in lrs:
#         for er_ep in er_episodes:
#             print(f'batch_size: {batch_size}, lr: {lr}, er_episodes: {er_ep}')
#             %run train_pcn.py --env=dilemma --starting_loc_x=4 --starting_loc_y=0 --batch_size={batch_size} --lr={lr} --num_er_episodes={er_ep} --timesteps=2000

# #%% Amsterdam Hypeparameter Search
# # batch_sizes = [128, 256, 512]
# # lrs = [1e-1, 1e-2, 1e-3, 1e-4]
# # er_episodes = [25, 50, 100]
# # This is not a hyperparameter, it's the number of objectives

# batch_sizes = [128, 256, 512]
# lrs = [1e-1, 1e-2, 1e-3, 1e-4]
# er_episodes = [25, 50, 100]

# nr_groups = 5

# for batch_size in batch_sizes:
#     for lr in lrs:
#         for er_ep in er_episodes:
#             print(f'batch_size: {batch_size}, lr: {lr}, er_episodes: {er_ep}')
#             %run train_pcn.py --env=amsterdam --starting_loc_x=9 --starting_loc_y=19 --batch_size={batch_size} --lr={lr} --nr_groups={nr_groups} --num_er_episodes={er_ep} --timesteps=30000
