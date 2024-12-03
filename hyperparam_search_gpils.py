import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import mo_gymnasium as mo_gym
import numpy as np
import wandb
import wandb.sdk
import yaml
from mo_gymnasium.utils import MORecordEpisodeStatistics

from morl_baselines.common.evaluation import seed_everything
from morl_baselines.common.experiments import (
    ALGOS,
    ENVS_WITH_KNOWN_PARETO_FRONT,
    StoreDict,
)
from morl_baselines.common.utils import reset_wandb_env
from motndp.city import City
from motndp.constraints import MetroConstraints

from morl_baselines.multi_policy.gpi_pd.gpi_pd_tndp import GPILS
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPILS as GPILSDST

from gymnasium.envs.registration import register

from morl_baselines.multi_policy.lcn.lcn import LCNTNDPModel
from morl_baselines.multi_policy.lcn.lcn_dst import LCNDST
register(
    id="motndp_dilemma-v0",
    entry_point="motndp.motndp:MOTNDP",
)

register(
    id="motndp_amsterdam-v0",
    entry_point="motndp.motndp:MOTNDP",
)

register(
    id="motndp_xian-v0",
    entry_point="motndp.motndp:MOTNDP",
)


import os
import yaml
import wandb

def train(seed, args, config):
    def make_env(gym_env):
        if gym_env == 'deep-sea-treasure-concave-v0':
            return mo_gym.make(gym_env)
        
        city = City(
            args.city_path,
            groups_file=args.groups_file,
            ignore_existing_lines=args.ignore_existing_lines
        )
        
        env = mo_gym.make(args.gym_env, 
                        city=city, 
                        constraints=MetroConstraints(city),
                        nr_stations=args.nr_stations,
                        state_representation='multi_binary')

        return env
    # Reset the wandb environment variables
    # reset_wandb_env()

    with wandb.init(project=args.project_name, config=config) as run:
        # Set the seed
        seed_everything(seed)

        if args.algo == 'gpi_ls_discrete':
            print(f": Seed {seed}. Instantiating {args.algo} on {args.env_id}")
            
            env = make_env(args.env_id)
            eval_env = make_env(args.env_id)

            # Launch the agent training
            print(f"Seed {seed}. Training agent...")
            
            if args.env_id == 'deep-sea-treasure-concave-v0':
                algo = GPILSDST(
                    env,
                    num_nets=1,
                    max_grad_norm=None,
                    gamma=1,
                    initial_epsilon=1.0,
                    final_epsilon=0.05,
                    alpha_per=0.6,
                    min_priority=0.01,
                    per=False,
                    use_gpi=True,
                    tau=1,
                    real_ratio=0.5,
                    log=True,
                    wandb_entity=args.wandb_entity,
                    **config,
                )
            else:
                algo = GPILS(
                    env,
                    num_nets=1,
                    max_grad_norm=None,
                    gamma=1,
                    initial_epsilon=1.0,
                    final_epsilon=0.05,
                    alpha_per=0.6,
                    min_priority=0.01,
                    per=False,
                    use_gpi=True,
                    tau=1,
                    real_ratio=0.5,
                    log=True,
                    experiment_name=args.experiment_name,
                    action_mask_dim=8,
                    wandb_entity=args.wandb_entity,
                    **config,
                )
            
            algo.train(
                total_timesteps=args.total_timesteps,
                ref_point=args.ref_point,
                eval_env=eval_env,
                weight_selection_algo='gpi-ls',
                num_eval_weights_for_front=args.num_eval_weights_for_front,
                timesteps_per_iter=args.timesteps_per_iter,
                eval_freq=args.eval_freq,
                eval_mo_freq=args.eval_mo_freq,
            )
        elif args.algo == 'lcn_dst':
            print(f": Seed {seed}. Instantiating {args.algo} on {args.env_id}")
            
            def make_env(gym_env):
                return mo_gym.make(gym_env)
            
            env = make_env(args.env_id)
            eval_env = make_env(args.env_id)

            # Launch the agent training
            print(f"Seed {seed}. Training agent...")
            
            algo = LCNDST(
                env,
                scaling_factor=args.scaling_factor,
                learning_rate=config['learning_rate'],
                batch_size=config['batch_size'],
                nr_layers=args.nr_layers,
                hidden_dim=config['hidden_dim'],
                distance_ref=args.distance_ref,
                lcn_lambda=args.lcn_lambda,
                seed=args.seed,
                model_class=LCNTNDPModel,
                wandb_entity=args.wandb_entity,
            )
            
            print(f"Seed {seed}. Training agent...")
            algo.train(
                total_timesteps=args.total_timesteps,
                eval_env=eval_env,
                ref_point=args.ref_point,
                num_er_episodes=config['num_er_episodes'],
                num_step_episodes=args.num_step_episodes,
                max_buffer_size=config['max_buffer_size'],
                num_model_updates=config['num_model_updates'],
                max_return=args.max_return,
                cd_threshold=args.cd_threshold,
            )
            
            
def main(args, seeds):
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
            train(seeds[0], args, config)

    # Start the sweep agent
    wandb.agent(sweep_id, function=sweep_wrapper, count=args.sweep_count, entity=args.wandb_entity, project=args.project_name)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, help="Name of the algorithm to run", choices=ALGOS.keys(), required=True)
    parser.add_argument("--env", type=str, help="environment", required=True)
    parser.add_argument("--wandb-entity", type=str, help="Wandb entity to use for the sweep", required=False)
    parser.add_argument("--project-name", type=str, help="Project name to use for the sweep", default="MORL-Baselines")
    parser.add_argument("--sweep-id", type=str, help="Sweep id to use if it already exists (helpful to parallelize the search)", required=False)
    parser.add_argument("--sweep-count", type=int, help="Number of trials to do in the sweep worker", default=10)
    parser.add_argument("--num-seeds", type=int, help="Number of seeds to use for the sweep", default=3)
    parser.add_argument("--seed", type=int, help="Random seed to start from, seeds will be in [seed, seed+num-seeds)", default=10)
    parser.add_argument("--config-path", type=str, help="path of config file.")
    parser.add_argument('--nr_groups', default=2, type=int)
    parser.add_argument('--distance_ref', default='nondominated', type=str, choices=['nondominated', 'optimal_max', 'nondominated_mean', 'interpolate', 'interpolate2', 'interpolate3'], help='controls the reference point for calculating the distance of every solution to the optimal point.')
    parser.add_argument('--lcn_lambda', default=None, type=float, help='value between 0 and 1. Controls the size of the front to explore. lambda -> 1: full pareto front. lambda -> 0 full lorenz front.')

    args = parser.parse_args()
    
    if args.env == 'dilemma':
        args.env_id = 'motndp_dilemma-v0'
        args.city_path = Path(f"./envs/mo-tndp/cities/dilemma_5x5")
        # args.total_timesteps = 3000
        args.total_timesteps = 500
        args.nr_stations = 9
        args.project_name = "MORL-TNDP"
        args.groups_file = "groups.txt"
        args.ignore_existing_lines = True
        args.experiment_name = "GPI-LS-Dilemma"
        args.ref_point = np.array([0, 0])
        args.starting_loc = (4, 0)
        args.num_eval_weights_for_front = 100
        args.timesteps_per_iter = 100
        args.eval_freq = 100
        args.eval_mo_freq = 100
    elif args.env == 'xian':
        args.env_id = 'motndp_xian-v0'
        args.city_path = Path(f"./envs/mo-tndp/cities/xian")
        args.total_timesteps = 30000
        args.nr_stations = 20
        args.project_name = "MORL-TNDP"
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.ignore_existing_lines = True
        args.experiment_name = "GPI-LS-Xian"
        args.ref_point = np.array([0] * args.nr_groups)
        args.starting_loc = (9, 19)
        args.timesteps_per_iter = 5000
        args.epsilon_decay_steps = 20000
        args.num_eval_weights_for_front = 100
        args.eval_freq = 5000
        args.eval_mo_freq = 5000
    elif args.env == 'amsterdam':
        args.env_id = 'motndp_amsterdam-v0'
        args.city_path = Path(f"./envs/mo-tndp/cities/amsterdam")
        args.total_timesteps = 30000
        args.nr_stations = 10
        args.project_name = "MORL-TNDP"
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.ignore_existing_lines = True
        args.experiment_name = "GPI-LS-Amsterdam"
        args.ref_point = np.array([0] * args.nr_groups)
        args.starting_loc = (9, 19)
        args.timesteps_per_iter = 5000
        args.epsilon_decay_steps = 20000
        args.num_eval_weights_for_front = 100
        args.eval_freq = 5000
        args.eval_mo_freq = 5000
    elif args.env == 'dst-gpils':
        args.env_id = 'deep-sea-treasure-concave-v0'
        args.total_timesteps = 100000
        args.nr_stations = 0
        args.project_name = "DST"
        args.experiment_name = "GPI-LS-DST"
        args.ref_point = np.array([0.0, -200.0])
        args.timesteps_per_iter = 5000
        args.epsilon_decay_steps = 80000
        args.num_eval_weights_for_front = 100
        args.eval_freq = 5000
        args.eval_mo_freq = 5000
    elif args.env == 'dst':
        args.env_id = 'deep-sea-treasure-concave-v0'
        args.project_name = "DST"
        args.experiment_name = "LCN-DST"
        args.total_timesteps = 100000
        args.nr_stations = 0
        args.nr_layers = 1
        # args.groups_file = f"price_groups_{args.nr_groups}.txt"
        # args.ignore_existing_lines = True
        # args.starting_loc = (9, 19)
        args.scaling_factor = np.array([0.1, 0.1, 0.01])
        args.ref_point = np.array([0.0, -200.0])
        args.max_return=np.array([124, -1])
        args.num_step_episodes = 10
        args.cd_threshold = .2
    
        
    # Create an array of seeds to use for the sweep
    seeds = [args.seed + i for i in range(args.num_seeds)]

    main(args, seeds)