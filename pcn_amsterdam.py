from pathlib import Path
import mo_gymnasium as mo_gym
from motndp.city import City
import numpy as np
from mo_gymnasium.utils import MORecordEpisodeStatistics
import envs
import argparse

from morl_baselines.multi_policy.pcn.pcn_tndp import PCNTNDP

def main(args):
    def make_env():
        city = City(
            Path(f"./envs/mo-tndp/cities/amsterdam"), 
            groups_file="price_groups_5.txt",
            ignore_existing_lines=True
        )
        
        env = mo_gym.make('motndp_amsterdam-v0', 
                        city=city, 
                        nr_stations=args.nr_stations)

        # env = MORecordEpisodeStatistics(env, gamma=1.0)
        return env

    env = make_env()

    agent = PCNTNDP(
        env,
        scaling_factor=np.array([1, 1, 1, 1, 1, 0.1]),
        learning_rate=1e-3,
        batch_size=256,
        project_name="MORL-TNDP",
        experiment_name="PCN-Amsterdam",
        log=not args.no_log,
    )

    agent.train(
        eval_env=make_env(),
        total_timesteps=int(10000),
        ref_point=np.array([0, 0, 0, 0, 0]),
        num_er_episodes=20,
        max_buffer_size=50,
        num_model_updates=50,
        starting_loc=(11, 15),
        # max_return=np.array([1.5, 1.5, -0.0]),
        # known_pareto_front=env.unwrapped.pareto_front(gamma=1.0),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MO PCN - TNDP")
    parser.add_argument('--no_log', action='store_true', default=False)
    # Episode horizon -- used as a proxy of both the budget and the number of stations (stations are not really costed)
    parser.add_argument('--nr_stations', default=20, type=int)

    args = parser.parse_args()
    main(args)
