#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=pcn_amsterdam
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

# kind-sweep-71
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=amsterdam --batch_size=128 --lr=0.01 --num_er_episodes=100 --max_buffer_size=100 
#     --num_model_updates=5 --hidden_dim=64 --starting_loc_x=9 --starting_loc_y=19 --nr_groups=2 --timesteps=30000 --nr_layers=1 --nr_stations=10 --seed=$SEED"

# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=amsterdam --batch_size=128 --lr=0.01 --num_er_episodes=100 --max_buffer_size=100 
#     --num_model_updates=5 --hidden_dim=64 --starting_loc_x=9 --starting_loc_y=19 --nr_groups=3 --timesteps=30000 --nr_layers=1 --nr_stations=10 --seed=$SEED"

# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=amsterdam --batch_size=256 --lr=0.1 --num_er_episodes=50 --max_buffer_size=50 
#     --num_model_updates=5 --hidden_dim=64 --starting_loc_x=9 --starting_loc_y=19 --nr_groups=4 --timesteps=30000 --nr_layers=1 --nr_stations=10 --seed=$SEED"

# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=amsterdam --batch_size=256 --lr=0.1 --num_er_episodes=50 --max_buffer_size=50 
#     --num_model_updates=5 --hidden_dim=64 --starting_loc_x=9 --starting_loc_y=19 --nr_groups=5 --timesteps=30000 --nr_layers=1 --nr_stations=10 --seed=$SEED"

CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=amsterdam --batch_size=256 --lr=0.1 --num_er_episodes=50 --max_buffer_size=50 
    --num_model_updates=5 --hidden_dim=64 --starting_loc_x=9 --starting_loc_y=19 --nr_groups=6 --timesteps=30000 --nr_layers=1 --nr_stations=10 --seed=$SEED"

# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=amsterdam --batch_size=256 --lr=0.1 --num_er_episodes=50 --max_buffer_size=50 
#     --num_model_updates=5 --hidden_dim=64 --starting_loc_x=9 --starting_loc_y=19 --nr_groups=7 --timesteps=30000 --nr_layers=1 --nr_stations=10 --seed=$SEED"

# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=amsterdam --batch_size=256 --lr=0.1 --num_er_episodes=50 --max_buffer_size=50 
# --num_model_updates=5 --hidden_dim=64 --starting_loc_x=9 --starting_loc_y=19 --nr_groups=8 --timesteps=30000 --nr_layers=1 --nr_stations=10 --seed=$SEED"

# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=amsterdam --batch_size=256 --lr=0.1 --num_er_episodes=50 --max_buffer_size=50 --num_model_updates=5 --hidden_dim=64 --starting_loc_x=9 --starting_loc_y=19 --nr_groups=9 --timesteps=30000 --nr_layers=1 --nr_stations=10 --seed=$SEED"

# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=amsterdam --batch_size=256 --lr=0.1 --num_er_episodes=50 --max_buffer_size=50 --num_model_updates=5 --hidden_dim=64 --starting_loc_x=9 --starting_loc_y=19 --nr_groups=10 --timesteps=30000 --nr_layers=1 --nr_stations=10 --seed=$SEED"


# Old State
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=amsterdam --batch_size=256 --lr=1e-2 --num_er_episodes=50 --max_buffer_size=50 --num_model_updates=5 --hidden_dim=128 --starting_loc_x=9 --starting_loc_y=19 --nr_groups=2 --timesteps=30000 --nr_layers=1 --nr_stations=10 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=amsterdam --batch_size=128 --lr=1e-2 --num_er_episodes=100 --max_buffer_size=100 --num_model_updates=5 --hidden_dim=64 --starting_loc_x=9 --starting_loc_y=19 --nr_groups=3 --timesteps=30000 --nr_layers=1 --nr_stations=10 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=amsterdam --batch_size=256 --lr=1e-1 --num_er_episodes=100 --max_buffer_size=50 --num_model_updates=5 --hidden_dim=64 --starting_loc_x=9 --starting_loc_y=19 --nr_groups=4 --timesteps=30000 --nr_layers=2 --nr_stations=10 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=amsterdam --batch_size=128 --lr=1e-1 --num_er_episodes=50 --max_buffer_size=50 --num_model_updates=5 --hidden_dim=64 --starting_loc_x=9 --starting_loc_y=19 --nr_groups=5 --timesteps=30000 --nr_layers=2 --nr_stations=10 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=amsterdam --batch_size=256 --lr=1e-2 --num_er_episodes=100 --max_buffer_size=50 --num_model_updates=5 --hidden_dim=64 --starting_loc_x=9 --starting_loc_y=19 --nr_groups=6 --timesteps=30000 --nr_layers=2 --nr_stations=10 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=amsterdam --batch_size=128 --lr=1e-2 --num_er_episodes=50 --max_buffer_size=50 --num_model_updates=5 --hidden_dim=64 --starting_loc_x=9 --starting_loc_y=19 --nr_groups=7 --timesteps=30000 --nr_layers=1 --nr_stations=10 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=amsterdam --batch_size=128 --lr=1e-2 --num_er_episodes=100 --max_buffer_size=100 --num_model_updates=5 --hidden_dim=128 --starting_loc_x=9 --starting_loc_y=19 --nr_groups=8 --timesteps=30000 --nr_layers=1 --nr_stations=10 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=amsterdam --batch_size=256 --lr=1e-1 --num_er_episodes=100 --max_buffer_size=50 --num_model_updates=5 --hidden_dim=64 --starting_loc_x=9 --starting_loc_y=19 --nr_groups=9 --timesteps=30000 --nr_layers=1 --nr_stations=10  --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_pcn.py --env=amsterdam --batch_size=128 --lr=1e-2 --num_er_episodes=50 --max_buffer_size=50 --num_model_updates=5 --hidden_dim=128 --starting_loc_x=9 --starting_loc_y=19 --nr_groups=10 --timesteps=30000 --nr_layers=2 --nr_stations=10  --seed=$SEED"

$CMD