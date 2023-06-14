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
            %run train_pcn.py --env=dilemma --batch_size={batch_size} --lr={lr} --num_er_episodes={er_ep} --timesteps=2000

#%% Amsterdam Hypeparameter Search