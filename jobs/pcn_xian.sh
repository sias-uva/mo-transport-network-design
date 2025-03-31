#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=pcn_xian
#SBATCH --partition=all
##SBATCH --partition=all6000
##SBATCH --account=all6000users
#SBATCH --gres=gpu:1
#SBATCH --mem=62G
#SBATCH --cpus-per-task=12
#SBATCH --time=10:00:00

if [ -z "$1" ]; then
    echo "Error: Missing required argument <seed>"
    echo "Usage: $0 <seed>"
    exit 1
fi

SEED=$1

# New State
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=2 --timesteps=30000 --batch_size=128 --hidden_dim=64 --lr=0.01 --max_buffer_size=50 --nr_layers=1 --nr_stations=20 --num_er_episodes=100 --num_model_updates=5 --num_step_episodes=10 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=3 --timesteps=30000 --batch_size=128 --hidden_dim=64 --lr=0.01 --max_buffer_size=50 --nr_layers=1 --nr_stations=20 --num_er_episodes=100 --num_model_updates=5 --num_step_episodes=10 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=4 --timesteps=30000 --batch_size=128 --hidden_dim=64 --lr=0.1 --max_buffer_size=100 --nr_layers=2 --nr_stations=20 --num_er_episodes=50 --num_model_updates=5 --num_step_episodes=10 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=5 --timesteps=30000 --batch_size=128 --hidden_dim=64 --lr=0.1 --max_buffer_size=100 --nr_layers=2 --nr_stations=20 --num_er_episodes=50 --num_model_updates=5 --num_step_episodes=10 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=6 --timesteps=30000 --batch_size=128 --hidden_dim=64 --lr=0.1 --max_buffer_size=100 --nr_layers=2 --nr_stations=20 --num_er_episodes=50 --num_model_updates=5 --num_step_episodes=10 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=7 --timesteps=30000 --batch_size=128 --hidden_dim=64 --lr=0.1 --max_buffer_size=100 --nr_layers=2 --nr_stations=20 --num_er_episodes=50 --num_model_updates=5 --num_step_episodes=10 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=8 --timesteps=30000 --batch_size=128 --hidden_dim=64 --lr=0.1 --max_buffer_size=100 --nr_layers=2 --nr_stations=20 --num_er_episodes=50 --num_model_updates=5 --num_step_episodes=10 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=9 --timesteps=30000 --batch_size=128 --hidden_dim=64 --lr=0.1 --max_buffer_size=100 --nr_layers=2 --nr_stations=20 --num_er_episodes=50 --num_model_updates=5 --num_step_episodes=10 --seed=$SEED"
CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=10 --timesteps=30000 --batch_size=128 --hidden_dim=64 --lr=0.1 --max_buffer_size=100 --nr_layers=2 --nr_stations=20 --num_er_episodes=50 --num_model_updates=5 --num_step_episodes=10 --seed=$SEED"

# Old State
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=2 --timesteps=30000 --batch_size=128 --hidden_dim=64 --lr=0.01 --max_buffer_size=50 --nr_layers=3 --nr_stations=20 --num_er_episodes=50 --num_model_updates=5 --num_step_episodes=10 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=3 --timesteps=30000 --batch_size=128 --hidden_dim=128 --lr=0.01 --max_buffer_size=100 --nr_layers=1 --nr_stations=20 --num_er_episodes=50 --num_model_updates=10 --num_step_episodes=10 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=4 --timesteps=30000 --batch_size=256 --hidden_dim=128 --lr=0.01 --max_buffer_size=50 --nr_layers=1 --nr_stations=20 --num_er_episodes=100 --num_model_updates=5 --num_step_episodes=10 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=5 --timesteps=30000 --batch_size=256 --hidden_dim=64 --lr=0.01 --max_buffer_size=100 --nr_layers=2 --nr_stations=20 --num_er_episodes=100 --num_model_updates=5 --num_step_episodes=10 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=6 --timesteps=30000 --batch_size=256 --hidden_dim=128 --lr=0.01 --max_buffer_size=50 --nr_layers=1 --nr_stations=20 --num_er_episodes=50 --num_model_updates=5 --num_step_episodes=10 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=7 --timesteps=30000 --batch_size=256 --hidden_dim=128 --lr=0.01 --max_buffer_size=50 --nr_layers=2 --nr_stations=20 --num_er_episodes=100 --num_model_updates=5 --num_step_episodes=10 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=8 --timesteps=30000 --batch_size=256 --hidden_dim=64 --lr=0.01 --max_buffer_size=100 --nr_layers=1 --nr_stations=20 --num_er_episodes=100 --num_model_updates=5 --num_step_episodes=10 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=9 --timesteps=30000 --batch_size=256 --hidden_dim=128 --lr=0.01 --max_buffer_size=100 --nr_layers=1 --nr_stations=20 --num_er_episodes=50 --num_model_updates=5 --num_step_episodes=10 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=10 --timesteps=30000 --batch_size=256 --hidden_dim=128 --lr=0.01 --max_buffer_size=50 --nr_layers=1 --nr_stations=20 --num_er_episodes=50 --num_model_updates=5 --num_step_episodes=10 --seed=$SEED"

$CMD