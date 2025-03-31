#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=lcn_nd_dst
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=62G
#SBATCH --cpus-per-task=12
#SBATCH --time=10:00:00

SEED=1234

CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_lcn.py --env=dst --batch_size=256 --lr=0.001 --num_er_episodes=500 --max_buffer_size=1000 --num_model_updates=10 --hidden_dim=128 --timesteps=100000 --distance_ref nondominated --nr_layers=1 --nr_stations=0 --seed=$SEED"

$CMD