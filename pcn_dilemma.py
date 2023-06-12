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
            Path(f"./envs/mo-tndp/cities/dilemma_5x5"), 
            groups_file="groups.txt",
            ignore_existing_lines=True
        )
        
        env = mo_gym.make('motndp_dilemma-v0', 
                        city=city, 
                        constraints=MetroConstraints(city),
                        nr_stations=args.nr_stations)

        # env = MORecordEpisodeStatistics(env, gamma=1.0)
        return env

    env = make_env()

    agent = PCNTNDP(
        env,
        scaling_factor=np.array([1, 1, 0.1]),
        learning_rate=args.lr,
        batch_size=args.batch_size,
        project_name="MORL-TNDP",
        experiment_name="PCN-Dilemma",
        log=not args.no_log,
    )

    save_dir = Path(f"./results/pcn_dilemma_{datetime.datetime.today().strftime('%Y%m%d_%H_%M_%S.%f')}")
    agent.train(
        eval_env=make_env(),
        total_timesteps=args.timesteps,
        ref_point=np.array([0, 0]),
        num_er_episodes=10,
        max_buffer_size=50,
        num_model_updates=50,
        starting_loc=(4, 0),
        max_return=np.array([1, 1]),
        save_dir=save_dir,
        # known_pareto_front=env.unwrapped.pareto_front(gamma=1.0),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MO PCN - TNDP")
    parser.add_argument('--no_log', action='store_true', default=False)
    # Episode horizon -- used as a proxy of both the budget and the number of stations (stations are not really costed)
    parser.add_argument('--nr_stations', default=9, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--timesteps', default=1000, type=int)

    args = parser.parse_args()
    main(args)
