#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=gpils_xian
##SBATCH --partition=all6000
##SBATCH --account=all6000users
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=62G
#SBATCH --cpus-per-task=12
#SBATCH --time=5:00:00

SEED=10

# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_gpipd.py --env xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=2 --total_timesteps=30000 --epsilon_decay_steps=20000 --timesteps_per_iter=5000 --nr_layers=3 --hidden_dim=64 --batch_size=64 --buffer_size=512 --learning_rate=0.0001 --target_update_freq=50 --gradient_updates=2 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_gpipd.py --env xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=3 --total_timesteps=30000 --epsilon_decay_steps=20000 --timesteps_per_iter=5000 --nr_layers=3 --hidden_dim=64 --batch_size=512 --buffer_size=8192 --learning_rate=0.00001 --target_update_freq=20 --gradient_updates=1 --seed=$SEED"
CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_gpipd.py --env xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=4 --total_timesteps=30000 --epsilon_decay_steps=20000 --timesteps_per_iter=5000 --nr_layers=3 --hidden_dim=128 --batch_size=16 --buffer_size=16384 --learning_rate=0.00001 --target_update_freq=10 --gradient_updates=1 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_gpipd.py --env xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=5 --total_timesteps=30000 --epsilon_decay_steps=20000 --timesteps_per_iter=5000 --nr_layers=3 --hidden_dim=128 --batch_size=32 --buffer_size=2048 --learning_rate=0.00001 --target_update_freq=10 --gradient_updates=1 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_gpipd.py --env xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=6 --total_timesteps=30000 --epsilon_decay_steps=20000 --timesteps_per_iter=5000 --nr_layers=3 --hidden_dim=128 --batch_size=16 --buffer_size=8192 --learning_rate=0.00001 --target_update_freq=50 --gradient_updates=2 --seed=$SEED"

$CMD