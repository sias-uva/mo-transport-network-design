#%% This notebook is meant to perform hyper-parameter search.
# It's a script that's not necessary to train and evaluate the PCN, but it's useful to find good hyper-parameters.

#%% Dilemma Hypeparameter Search
batch_sizes = [128, 256, 512]
lrs = [1e-1, 1e-2, 1e-3, 1e-4]
er_episodes = [25, 50, 100]

for batch_size in batch_sizes:
    for lr in lrs:
        for er_ep in er_episodes:
            print(f'batch_size: {batch_size}, lr: {lr}, er_episodes: {er_ep}')
            %run train_pcn.py --env=dilemma --starting_loc_x=4 --starting_loc_y=0 --batch_size={batch_size} --lr={lr} --num_er_episodes={er_ep} --timesteps=2000

#%% Amsterdam Hypeparameter Search
batch_sizes = [128, 256, 512]
lrs = [1e-1, 1e-2, 1e-3, 1e-4]
er_episodes = [25, 50, 100]
# This is not a hyperparameter, it's the number of objectives
nr_groups = 5

for batch_size in batch_sizes:
    for lr in lrs:
        for er_ep in er_episodes:
            print(f'batch_size: {batch_size}, lr: {lr}, er_episodes: {er_ep}')
            %run train_pcn.py --env=amsterdam --starting_loc_x=9 --starting_loc_y=19 --batch_size={batch_size} --lr={lr} --nr_groups={nr_groups} --num_er_episodes={er_ep} --timesteps=10000
