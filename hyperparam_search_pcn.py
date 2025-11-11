## OUTDATED FILE, USE hyperparam_search_gpils.py INSTEAD
# It's a script that's not necessary to train and evaluate the PCN, but it's useful to perform grid-search on hyper-parameters.
import argparse
from itertools import product
import os
from pathlib import Path
import random

from motndp.city import City
from motndp.constraints import MetroConstraints
import numpy as np
import wandb
import yaml
from morl_baselines.multi_policy.pcn.pcn_tndp import PCNTNDP, PCNTNDPModel
from morl_baselines.multi_policy.lcn.lcn import LCNTNDP, LCNTNDPModel
from train_pcn import main as pcn_main
from train_lcn import main as lcn_main
import time
import torch
import mo_gymnasium as mo_gym

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)    
    torch.manual_seed(args.seed)

def train(seed, args, config):
    def make_env(gym_env):
        city = City(
            args.city_path,
            groups_file=args.groups_file,
            ignore_existing_lines=args.ignore_existing_lines
        )
        
        env = mo_gym.make(gym_env,
                        city=city, 
                        constraints=MetroConstraints(city),
                        nr_stations=args.nr_stations,
                        state_representation='dictionary',
                        od_type=args.od_type,
                        chained_reward=args.chained_reward,)

        return env

    with wandb.init(project=args.project_name, config=config) as run:
        # Set the seed
        seed_everything(seed)

        env = make_env(args.env_id)

        # Launch the agent training
        print(f"Seed {seed}. Training agent...")
        
        if args.model == "PCN":
            agent = PCNTNDP(
                env,
                scaling_factor=args.scaling_factor,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                nr_layers=config.nr_layers,
                hidden_dim=config.hidden_dim,
                project_name="MORL-TNDP",
                experiment_name=args.experiment_name,
                log=not args.no_log,
                seed=args.seed,
                model_class=PCNTNDPModel,
                wandb_entity=args.wandb_entity,
            )
        elif args.model == "LCN":
            agent = LCNTNDP(
                env,
                scaling_factor=args.scaling_factor,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                nr_layers=config.nr_layers,
                hidden_dim=config.hidden_dim,
                project_name="MORL-TNDP",
                experiment_name=args.experiment_name,
                log=not args.no_log,
                seed=args.seed,
                model_class=LCNTNDPModel,
                distance_ref=args.distance_ref,
                lcn_lambda=args.lcn_lambda,
                wandb_entity=args.wandb_entity,
            )
            
        agent.train(
            total_timesteps=config.total_timesteps,
            eval_env=make_env(args.env_id),
            ref_point=args.ref_point,
            num_er_episodes=config.num_er_episodes,
            num_step_episodes=config.num_step_episodes,
            max_buffer_size=config.max_buffer_size,
            num_model_updates=config.num_model_updates,
            starting_loc=args.starting_loc,
            nr_stations=args.nr_stations,
            max_return=args.max_return,
            cd_threshold=args.cd_threshold,
            n_policies=args.num_policies,
        )

def main(args, seed):
    config_file = os.path.join(args.config_path)

    # Set up the default hyperparameters
    with open(config_file) as file:
        sweep_config = yaml.load(file, Loader=yaml.FullLoader)

    # Set up the sweep -- if a sweep id is provided, use it, otherwise create a new sweep
    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        sweep_id = wandb.sweep(sweep=sweep_config, entity=args.wandb_entity, project=args.project_name)

    # Define a wrapper function for wandb.agent
    def sweep_wrapper():
        # Initialize a new wandb run
        with wandb.init() as run:
            # Get the current configuration
            config = run.config
            # Call the train function with the current configuration
            train(seed, args, config)

    # Start the sweep agent
    wandb.agent(sweep_id, function=sweep_wrapper, count=args.sweep_count, entity=args.wandb_entity, project=args.project_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"MO-TNDP Hyperparameter Search")
    # Acceptable values: 'dilemma', 'amsterdam'
    parser.add_argument('--model', default=None, type=str) # PCN or LCN
    parser.add_argument('--env', default='dilemma', type=str)
    parser.add_argument("--wandb-entity", type=str, help="Wandb entity to use for the sweep", required=False)
    parser.add_argument("--sweep-id", type=str, help="Sweep id to use if it already exists (helpful to parallelize the search)", required=False)
    parser.add_argument("--sweep-count", type=int, help="Number of trials to do in the sweep worker", default=100)
    parser.add_argument("--config-path", type=str, help="path of config file.")
    # For amsterdam environment we have different groups files (different nr of objectives)
    parser.add_argument('--nr_groups', default=5, type=int)
    parser.add_argument('--num_policies', default=10, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--no_log', action='store_true', default=False)
    parser.add_argument('--random', action='store_true', default=False, help='if true, the hyperparameter search will be random instead of exhaustive')
    parser.add_argument('--distance_ref', default='nondominated', type=str, choices=['nondominated', 'optimal_max', 'nondominated_mean', 'interpolate', 'interpolate2', 'interpolate3'], help='controls the reference point for calculating the distance of every solution to the optimal point.')
    parser.add_argument('--lcn_lambda', default=None, type=float)

    args = parser.parse_args()
    
    args.project_name = "MORL-TNDP"
    args.ignore_existing_lines = True
    args.od_type = "pct"
    args.chained_reward = True
    args.lcn_lambda = args.lcn_lambda
    args.distance_ref = args.distance_ref

    if args.env == 'amsterdam':
        args.env_id = 'motndp_amsterdam-v0'
        args.city_path = Path(f"./envs/mo-tndp/cities/amsterdam")
        args.experiment_name = "PCN-Amsterdam"
        args.nr_stations = 10
        args.gym_env = 'motndp_amsterdam-v0'
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.scaling_factor = np.array([100] * args.nr_groups + [0.01])
        args.ref_point = np.array([0] * args.nr_groups)
        args.max_return=np.array([1] * args.nr_groups)
        args.cd_threshold = 0.2
        args.starting_loc = (9, 19)
    elif args.env == 'xian':
        args.env_id = 'motndp_xian-v0'
        args.city_path = Path(f"./envs/mo-tndp/cities/xian")
        args.experiment_name = "PCN-Xian"
        args.nr_stations = 20
        args.gym_env = 'motndp_xian-v0'
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.scaling_factor = np.array([100] * args.nr_groups + [0.01])
        args.ref_point = np.array([0] * args.nr_groups)
        args.max_return=np.array([1] * args.nr_groups)
        args.cd_threshold = 0.2
        args.starting_loc = (9, 19)
    
    seed = 42

    main(args, seed)
