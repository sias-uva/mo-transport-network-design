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
from morl_baselines.multi_policy.lcn.lcn import LCNTNDP

def main(args):
    def make_env():
        city = City(
            args.city_path, 
            groups_file=args.groups_file,
            ignore_existing_lines=args.ignore_existing_lines
        )
        
        env = mo_gym.make(args.gym_env, 
                        city=city, 
                        constraints=MetroConstraints(city),
                        nr_stations=args.nr_stations)

        return env

    env = make_env()

    agent = LCNTNDP(
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
    )

    if args.starting_loc is None:
        print('NOTE: Training is running with random starting locations.')

    save_dir = Path(f"./results/lcn_{args.env}_{datetime.datetime.today().strftime('%Y%m%d_%H_%M_%S.%f')}")
    agent.train(
        eval_env=make_env(),
        total_timesteps=args.timesteps,
        ref_point=args.ref_point,
        num_er_episodes=args.num_er_episodes,
        num_explore_episodes=args.num_explore_episodes,
        num_step_episodes=args.num_step_episodes,
        max_buffer_size=args.max_buffer_size,
        num_model_updates=args.num_model_updates,
        starting_loc=args.starting_loc,
        nr_stations=args.nr_stations,
        max_return=args.max_return,
        save_dir=save_dir,
        pf_plot_limits=args.pf_plot_limits,
        n_policies=args.num_policies,
        train_mode=args.train_mode,
        update_interval=args.update_interval,
        cd_threshold=args.cd_threshold,
        # known_pareto_front=env.unwrapped.pareto_front(gamma=1.0),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MO LCN - TNDP")
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
    parser.add_argument('--num_explore_episodes', type=int, default=None, help='the nr of top episodes to use to calcualte the desired return when exploring. If None, it will use all ER episodes.')
    parser.add_argument('--num_step_episodes', default=10, type=int)
    parser.add_argument('--num_model_updates', default=10, type=int)
    parser.add_argument('--num_policies', default=10, type=int)
    parser.add_argument('--max_buffer_size', default=50, type=int)
    parser.add_argument('--nr_layers', default=1, type=int)
    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('--timesteps', default=2000, type=int)
    parser.add_argument('--no_log', action='store_true', default=False)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--train_mode', default='uniform', type=str, choices=['uniform', 'disttofront', 'disttofront2'], help='controls how to select episodes from the replay buffer for training. If uniform, episodes are sampled uniformly. If disttofront, episodes are sampled with probability proportional to their distance to the pareto front.')
    parser.add_argument('--update_interval', default=None, type=int, help='controls how often to update the model. If None, it will update every loop. If a number, it will update every update_interval steps.')
    parser.add_argument('--cd_threshold', default=0.2, type=float, help='controls the threshold for crowdedness distance.')

    args = parser.parse_args()

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
        args.experiment_name = "LCN-Dilemma"
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
        args.experiment_name = "LCN-Margins"
        args.scaling_factor = np.array([1, 1, 0.1])
        args.ref_point = np.array([0, 0])
        args.max_return=np.array([1, 1])
        args.pf_plot_limits = [0, 0.5]
    elif args.env == 'amsterdam':
        args.city_path = Path(f"./envs/mo-tndp/cities/amsterdam")
        args.gym_env = 'motndp_amsterdam-v0'
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.ignore_existing_lines = True
        args.experiment_name = "LCN-Amsterdam"
        args.scaling_factor = np.array([100] * args.nr_groups + [0.01])
        args.ref_point = np.array([0] * args.nr_groups)
        args.max_return=np.array([1] * args.nr_groups)
        args.pf_plot_limits = None
    
    if args.starting_loc_x is not None and args.starting_loc_y is not None:
        args.starting_loc = (args.starting_loc_x, args.starting_loc_y)
    else:
        args.starting_loc = None

    main(args)