#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=pcn_dst
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=62G
#SBATCH --cpus-per-task=12
#SBATCH --time=10:00:00

SEED=1234

CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=dst --batch_size=256 --lr=1e-2 --num_er_episodes=50 --max_buffer_size=200 --num_model_updates=10 --hidden_dim=64 --timesteps=100000 --nr_layers=1 --nr_stations=0 --seed=$SEED"

$CMD