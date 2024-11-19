import datetime
from pathlib import Path
import random
import mo_gymnasium as mo_gym
from motndp.city import City
from motndp.constraints import MetroConstraints
import numpy as np
import torch
import envs
import argparse
from morl_baselines.multi_policy.pcn.pcn import PCN, BasePCNModel, DiscreteActionsDefaultModel
from morl_baselines.multi_policy.pcn.pcn_tndp import PCNTNDP, PCNTNDPModel

def main(args):
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
                        state_representation='one_hot')

        return env

    env = make_env(args.gym_env)
    
    if args.gym_env == 'deep-sea-treasure-concave-v0':
        agent = PCN(
            env,
            scaling_factor=args.scaling_factor,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            project_name=args.project_name,
            experiment_name=args.experiment_name,
            log=not args.no_log,
            seed=args.seed,
            hidden_dim=args.hidden_dim,
            model_class=DiscreteActionsDefaultModel
        )
    else:
        agent = PCNTNDP(
            env,
            scaling_factor=args.scaling_factor,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            project_name="MORL-TNDP",
            experiment_name=args.experiment_name,
            log=not args.no_log,
            seed=args.seed,
            nr_layers=args.nr_layers,
            hidden_dim=args.hidden_dim,
            model_class=PCNTNDPModel
        )

        if args.starting_loc is None:
            print('NOTE: Training is running with random starting locations.')

    save_dir = Path(f"./results/pcn_{args.env}_{datetime.datetime.today().strftime('%Y%m%d_%H_%M_%S.%f')}")
    if args.gym_env == 'deep-sea-treasure-concave-v0':
        agent.train(
            total_timesteps=args.timesteps,
            eval_env=make_env(args.gym_env),
            ref_point=args.ref_point,
            num_er_episodes=args.num_er_episodes,
            num_step_episodes=args.num_step_episodes,
            max_buffer_size=args.max_buffer_size,
            num_model_updates=args.num_model_updates,
            max_return=args.max_return,
        )
    else:
        agent.train(
            total_timesteps=args.timesteps,
            eval_env=make_env(args.gym_env),
            ref_point=args.ref_point,
            num_er_episodes=args.num_er_episodes,
            num_step_episodes=args.num_step_episodes,
            max_buffer_size=args.max_buffer_size,
            num_model_updates=args.num_model_updates,
            starting_loc=args.starting_loc,
            nr_stations=args.nr_stations,
            max_return=args.max_return,
            save_dir=save_dir,
            pf_plot_limits=args.pf_plot_limits,
            n_policies=args.num_policies,
            cd_threshold=args.cd_threshold,
            # known_pareto_front=env.unwrapped.pareto_front(gamma=1.0),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MO PCN - TNDP")
    # Acceptable values: 'dilemma', 'margins', 'amsterdam'
    parser.add_argument('--env', default='dilemma', type=str)
    # For amsterdam environment we have different groups files (different nr of objectives)
    parser.add_argument('--nr_groups', default=5, type=int)
    # Starting location of the agent
    parser.add_argument('--starting_loc_x', default=None, type=int)
    parser.add_argument('--starting_loc_y', default=None, type=int)
    # Episode horizon -- used as a proxy of both the budget and the number of stations (stations are not really costed)
    parser.add_argument('--nr_stations', type=int, required=True)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_er_episodes', default=50, type=int)
    parser.add_argument('--num_step_episodes', default=10, type=int)
    parser.add_argument('--num_model_updates', default=10, type=int)
    parser.add_argument('--num_policies', default=10, type=int)
    parser.add_argument('--max_buffer_size', default=50, type=int)
    parser.add_argument('--nr_layers', default=1, type=int)
    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('--timesteps', default=2000, type=int)
    parser.add_argument('--no_log', action='store_true', default=False)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--cd_threshold', default=0.2, type=float, help='controls the threshold for crowdedness distance.')

    args = parser.parse_args()
    print(args)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Some values are hardcoded for each environment (this is flexible, but we don't want to have to pass 100 arguments to the script)
    if args.env == 'dilemma':
        args.city_path = Path(f"./envs/mo-tndp/cities/dilemma_5x5")
        args.nr_stations = 9
        args.gym_env = 'motndp_dilemma-v0'
        args.groups_file = "groups.txt"
        args.ignore_existing_lines = True
        args.experiment_name = "PCN-Dilemma"
        args.scaling_factor = np.array([1, 1, 0.1])
        args.ref_point = np.array([0, 0])
        args.max_return=np.array([1, 1])
        args.pf_plot_limits = [0, 0.5]
    elif args.env == 'margins':
        args.city_path = Path(f"./envs/mo-tndp/cities/margins_5x5")
        args.nr_stations = 9
        args.gym_env = 'motndp_margins-v0'
        args.groups_file = f"groups.txt"
        args.ignore_existing_lines = True
        args.experiment_name = "PCN-Margins"
        args.scaling_factor = np.array([1, 1, 0.1])
        args.ref_point = np.array([0, 0])
        args.max_return=np.array([1, 1])
        args.pf_plot_limits = [0, 0.5]
    elif args.env == 'amsterdam':
        args.city_path = Path(f"./envs/mo-tndp/cities/amsterdam")
        args.gym_env = 'motndp_amsterdam-v0'
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.ignore_existing_lines = True
        args.experiment_name = "PCN-Amsterdam"
        args.scaling_factor = np.array([100] * args.nr_groups + [0.01])
        args.ref_point = np.array([0] * args.nr_groups)
        args.max_return=np.array([1] * args.nr_groups)
        # args.pf_plot_limits = [0, 0.015]
        args.pf_plot_limits = None
    elif args.env == 'xian':
        args.city_path = Path(f"./envs/mo-tndp/cities/xian")
        args.gym_env = 'motndp_xian-v0'
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.ignore_existing_lines = True
        args.experiment_name = "PCN-Xian"
        args.scaling_factor = np.array([100] * args.nr_groups + [0.01])
        args.ref_point = np.array([0] * args.nr_groups)
        args.max_return=np.array([1] * args.nr_groups)
        args.pf_plot_limits = None
    elif args.env == 'amsterdam_10x10':
        args.city_path = Path(f"./envs/mo-tndp/cities/amsterdam_10x10")
        args.gym_env = 'motndp_amsterdam_10x10-v0'
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.ignore_existing_lines = True
        args.experiment_name = "PCN-Amsterdam_10x10"
        args.scaling_factor = np.array([100] * args.nr_groups + [0.01])
        args.ref_point = np.array([0] * args.nr_groups)
        args.max_return=np.array([1] * args.nr_groups)
        # args.pf_plot_limits = [0, 0.015]
        args.pf_plot_limits = None
    elif args.env == 'dst':
        args.gym_env = 'deep-sea-treasure-concave-v0'
        args.project_name = "DST"
        args.experiment_name = "PCN-DST"
        args.scaling_factor = np.array([0.1, 0.1, 0.01])
        args.ref_point = np.array([0.0, -200.0])
        args.max_return=np.array([124, -1])
        args.pf_plot_limits = None
    
    if args.starting_loc_x is not None and args.starting_loc_y is not None:
        args.starting_loc = (args.starting_loc_x, args.starting_loc_y)
    else:
        args.starting_loc = None

    main(args)
