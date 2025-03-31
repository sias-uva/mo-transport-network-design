#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=lambda_lcn_xian
#SBATCH --partition=all6000
#SBATCH --account=all6000users
#SBATCH --gres=gpu:1
#SBATCH --mem=62G
#SBATCH --cpus-per-task=12
#SBATCH --time=5:00:00

SEED=6982

CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_lcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=3 --timesteps=30000 --batch_size=256 --hidden_dim=64 --lr=0.01 --max_buffer_size=50 --nr_layers=1 --nr_stations=20 --num_er_episodes=50 --num_model_updates=5 --num_step_episodes=10 --distance_ref=interpolate3 --lcn_lambda=0.0 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_lcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=3 --timesteps=30000 --batch_size=256 --hidden_dim=64 --lr=0.01 --max_buffer_size=50 --nr_layers=2 --nr_stations=20 --num_er_episodes=100 --num_model_updates=5 --num_step_episodes=10 --distance_ref=interpolate3 --lcn_lambda=0.1 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_lcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=3 --timesteps=30000 --batch_size=128 --hidden_dim=128 --lr=0.01 --max_buffer_size=50 --nr_layers=2 --nr_stations=20 --num_er_episodes=50 --num_model_updates=5 --num_step_episodes=10 --distance_ref=interpolate3 --lcn_lambda=0.2 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_lcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=3 --timesteps=30000 --batch_size=256 --hidden_dim=128 --lr=0.1 --max_buffer_size=50 --nr_layers=1 --nr_stations=20 --num_er_episodes=100 --num_model_updates=5 --num_step_episodes=10 --distance_ref=interpolate3 --lcn_lambda=0.3 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_lcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=3 --timesteps=30000 --batch_size=128 --hidden_dim=128 --lr=0.01 --max_buffer_size=50 --nr_layers=1 --nr_stations=20 --num_er_episodes=100 --num_model_updates=5 --num_step_episodes=10 --distance_ref=interpolate3 --lcn_lambda=0.4 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_lcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=3 --timesteps=30000 --batch_size=256 --hidden_dim=128 --lr=0.1 --max_buffer_size=50 --nr_layers=1 --nr_stations=20 --num_er_episodes=100 --num_model_updates=5 --num_step_episodes=10 --distance_ref=interpolate3 --lcn_lambda=0.5 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_lcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=3 --timesteps=30000 --batch_size=256 --hidden_dim=128 --lr=0.01 --max_buffer_size=100 --nr_layers=1 --nr_stations=20 --num_er_episodes=100 --num_model_updates=5 --num_step_episodes=10 --distance_ref=interpolate3 --lcn_lambda=0.6 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_lcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=3 --timesteps=30000 --batch_size=128 --hidden_dim=64 --lr=0.01 --max_buffer_size=100 --nr_layers=1 --nr_stations=20 --num_er_episodes=50 --num_model_updates=5 --num_step_episodes=10 --distance_ref=interpolate3 --lcn_lambda=0.7 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_lcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=3 --timesteps=30000 --batch_size=128 --hidden_dim=128 --lr=0.01 --max_buffer_size=100 --nr_layers=1 --nr_stations=20 --num_er_episodes=100 --num_model_updates=5 --num_step_episodes=10 --distance_ref=interpolate3 --lcn_lambda=0.8 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_lcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=3 --timesteps=30000 --batch_size=256 --hidden_dim=128 --lr=0.1 --max_buffer_size=100 --nr_layers=1 --nr_stations=20 --num_er_episodes=100 --num_model_updates=5 --num_step_episodes=10 --distance_ref=interpolate3 --lcn_lambda=0.9 --seed=$SEED"
# CMD="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python train_lcn.py --env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=3 --timesteps=30000 --batch_size=128 --hidden_dim=128 --lr=0.1 --max_buffer_size=50 --nr_layers=1 --nr_stations=20 --num_er_episodes=50 --num_model_updates=5 --num_step_episodes=10 --distance_ref=interpolate3 --lcn_lambda=1.0 --seed=$SEED"

$CMD