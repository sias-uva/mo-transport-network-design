#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=gpils_dst
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=62G
#SBATCH --cpus-per-task=12
#SBATCH --time=8:00:00

SEED=6982

CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_gpipd.py --env dst --nr_stations=0 --total_timesteps=200000 --epsilon_decay_steps=150000 --timesteps_per_iter=5000 --nr_layers=3 --hidden_dim=128 --batch_size=512 --buffer_size=256 --learning_rate=0.0001 --target_update_freq=100 --gradient_updates=2 --seed=$SEED"

$CMD