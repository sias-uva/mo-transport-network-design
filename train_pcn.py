import datetime
from pathlib import Path
import mo_gymnasium as mo_gym
from motndp.city import City
from motndp.constraints import MetroConstraints
import numpy as np
from mo_gymnasium.utils import MORecordEpisodeStatistics
import envs
import argparse

from morl_baselines.multi_policy.pcn.pcn_tndp import PCNTNDP

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

    agent = PCNTNDP(
        env,
        scaling_factor=args.scaling_factor,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        project_name="MORL-TNDP",
        experiment_name=args.experiment_name,
        log=not args.no_log,
    )

    save_dir = Path(f"./results/pcn_{args.env}_{datetime.datetime.today().strftime('%Y%m%d_%H_%M_%S.%f')}")
    agent.train(
        eval_env=make_env(),
        total_timesteps=args.timesteps,
        ref_point=args.ref_point,
        num_er_episodes=args.num_er_episodes,
        max_buffer_size=args.max_buffer_size,
        num_model_updates=args.num_model_updates,
        starting_loc=args.starting_loc,
        max_return=args.max_return,
        save_dir=save_dir,
        # known_pareto_front=env.unwrapped.pareto_front(gamma=1.0),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MO PCN - TNDP")
    # Acceptable values: 'dilemma', 'amsterdam'
    parser.add_argument('--env', default='dilemma', type=str)
    parser.add_argument('--no_log', action='store_true', default=False)
    # Episode horizon -- used as a proxy of both the budget and the number of stations (stations are not really costed)
    # parser.add_argument('--nr_stations', default=9, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--timesteps', default=2000, type=int)

    args = parser.parse_args()

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
        args.num_er_episodes=50
        args.max_buffer_size=50
        args.num_model_updates=10
        args.starting_loc=(4, 0)
        args.max_return=np.array([1, 1])
    elif args.env == 'amsterdam':
        args.city_path = Path(f"./envs/mo-tndp/cities/amsterdam")
        args.nr_stations = 20
        args.gym_env = 'motndp_amsterdam-v0'
        args.groups_file = "price_groups_5.txt"
        args.ignore_existing_lines = True
        args.experiment_name = "PCN-Amsterdam"
        args.scaling_factor = np.array([1, 1, 1, 1, 1, 0.1])
        args.ref_point = np.array([0, 0, 0, 0, 0])
        args.num_er_episodes=50
        args.max_buffer_size=50
        args.num_model_updates=10
        args.starting_loc=(11, 14)
        args.max_return=np.array([1, 1, 1, 1, 1])
    
    main(args)
