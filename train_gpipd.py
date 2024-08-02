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
from morl_baselines.multi_policy.gpi_pd.gpi_pd_tndp import GPILS

def main(args):
    def make_env(gym_env):
        if gym_env == 'dst':
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
                        starting_loc=args.starting_loc,
                        obs_type='location_vector')

        return env

    env = make_env(args.gym_env)
    eval_env = make_env(args.gym_env)
    
    agent = GPILS(
        env,
        num_nets=1,
        max_grad_norm=None,
        learning_rate=args.learning_rate,
        gamma=1,
        batch_size=args.batch_size,
        net_arch=args.net_arch,
        buffer_size=int(args.buffer_size),
        initial_epsilon=1.0,
        final_epsilon=0.05,
        epsilon_decay_steps=args.epsilon_decay_steps,
        learning_starts=args.learning_starts,
        alpha_per=0.6,
        min_priority=0.01,
        per=False,
        use_gpi=True,
        gradient_updates=args.gradient_updates,
        target_net_update_freq=args.target_update_freq,
        tau=1,
        real_ratio=0.5,
        log=True,
        project_name="MORL-TNDP",
        experiment_name=args.experiment_name,
        action_mask_dim=8
    )
    
    if args.starting_loc is None:
        print('NOTE: Training is running with random starting locations.')

    save_dir = Path(f"./results/gpi_{args.env}_{datetime.datetime.today().strftime('%Y%m%d_%H_%M_%S.%f')}")
    
    agent.train(
        total_timesteps=args.total_timesteps,
        eval_env=eval_env,
        ref_point=args.ref_point,
        num_eval_weights_for_front=args.num_eval_weights_for_front,
        # known_pareto_front=env.unwrapped.pareto_front(gamma=0.98),
        weight_selection_algo='gpi-ls',
        timesteps_per_iter=args.timesteps_per_iter,
        eval_freq=args.eval_freq,
        eval_mo_freq=args.eval_mo_freq
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MO GPI-LS - TNDP")
    # Acceptable values: 'dilemma', 'margins', 'amsterdam', 'dst'
    parser.add_argument('--env', default='dilemma', type=str)
    # For amsterdam environment we have different groups files (different nr of objectives)
    parser.add_argument('--nr_groups', default=5, type=int)
    # Starting location of the agent
    parser.add_argument('--starting_loc_x', default=None, type=int)
    parser.add_argument('--starting_loc_y', default=None, type=int)
    # Episode horizon -- used as a proxy of both the budget and the number of stations (stations are not really costed)
    parser.add_argument('--nr_stations', type=int, required=True)
    parser.add_argument('--seed', default=42, type=int)

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
        args.project_name = "MORL-TNDP"
        args.groups_file = "groups.txt"
        args.ignore_existing_lines = True
        args.experiment_name = "GPI-LS-Dilemma"
        args.ref_point = np.array([0, 0])
        args.max_return=np.array([1, 1])
        args.pf_plot_limits = [0, 0.5]
        args.net_arch = [64, 64]
        args.batch_size = 32
        args.timesteps_per_iter = 100
        args.total_timesteps = 3000
        args.epsilon_decay_steps = 1500
        args.num_eval_weights_for_front = 100
        args.buffer_size = 512
        args.learning_starts = 50
        args.learning_rate = 1e-5
        args.target_update_freq = 100
        args.gradient_updates = 5
        args.eval_freq = 100
        args.eval_mo_freq = 100
    elif args.env == 'margins':
        args.city_path = Path(f"./envs/mo-tndp/cities/margins_5x5")
        args.nr_stations = 9
        args.gym_env = 'motndp_margins-v0'
        args.project_name = "MORL-TNDP"
        args.groups_file = f"groups.txt"
        args.ignore_existing_lines = True
        args.experiment_name = "GPI-LS-Margins"
        args.ref_point = np.array([0, 0])
        args.max_return=np.array([1, 1])
        args.pf_plot_limits = [0, 0.5]
    elif args.env == 'amsterdam':
        args.city_path = Path(f"./envs/mo-tndp/cities/amsterdam")
        args.gym_env = 'motndp_amsterdam-v0'
        args.project_name = "MORL-TNDP"
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.ignore_existing_lines = True
        args.experiment_name = "GPI-LS-Amsterdam"
        args.ref_point = np.array([0] * args.nr_groups)
        args.max_return=np.array([1] * args.nr_groups)
        args.pf_plot_limits = None
    # if args.env == 'amsterdam_10x10':
    #     args.city_path = Path(f"./envs/mo-tndp/cities/amsterdam_10x10")
    #     args.nr_stations = 10
    #     args.gym_env = 'motndp_amsterdam_10x10-v0'
    #     args.project_name = "MORL-TNDP"
    #     args.groups_file = "groups.txt"
    #     args.ignore_existing_lines = True
    #     args.experiment_name = "GPI-LS-Amsterdam_10x10"
    #     args.ref_point = np.array([0, 0])
    #     args.max_return=np.array([1, 1])
    #     args.pf_plot_limits = [0, 0.5]
    #     args.net_arch = [64, 64]
    #     args.batch_size = 32
    #     args.timesteps_per_iter = 100
    #     args.total_timesteps = 3000
    #     args.epsilon_decay_steps = 1500
    #     args.num_eval_weights_for_front = 100
    #     args.buffer_size = 5000
    #     args.learning_starts = 100
    #     args.learning_rate = 1e-5
    #     args.target_update_freq = 100
    #     args.gradient_updates = 1
    elif args.env == 'xian':
        args.city_path = Path(f"./envs/mo-tndp/cities/xian")
        args.gym_env = 'motndp_xian-v0'
        args.project_name = "MORL-TNDP"
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.ignore_existing_lines = True
        args.experiment_name = "GPI-LS-Xian"
        args.ref_point = np.array([0] * args.nr_groups)
        args.max_return=np.array([1] * args.nr_groups)
        args.pf_plot_limits = None
        args.net_arch = [256, 256, 256, 256]
        args.batch_size = 128
        args.timesteps_per_iter = 10000
        args.total_timesteps = 200000
        args.epsilon_decay_steps = 100000
        args.num_eval_weights_for_front = 100
        args.buffer_size = 1000000
        args.learning_starts = 100
        args.learning_rate = 3e-4
        args.target_update_freq = 200
        args.gradient_updates = 10
        args.eval_freq = 10000
        args.eval_mo_freq = 10000
    elif args.env == 'dst':
        args.gym_env = 'DeepSeaTreasure-v0'
        args.project_name = "DST"
        args.experiment_name = "GPI-LS-DST"
        args.ref_point = np.array([0, 0])
        args.max_return=np.array([1, 1])
        args.pf_plot_limits = [0, 0.5]

    if args.starting_loc_x is not None and args.starting_loc_y is not None:
        args.starting_loc = (args.starting_loc_x, args.starting_loc_y)
    else:
        args.starting_loc = None
    
    main(args)
