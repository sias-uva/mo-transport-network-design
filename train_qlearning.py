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

from morl_baselines.common.utils import linearly_decaying_value

class QLearningTNDP:
    def __init__(
        self, 
        env, 
        alpha: float,
        gamma: float,  
        initial_epsilon, 
        final_epsilon, 
        epsilon_decay_steps, 
        train_episodes, 
        test_episodes, 
        nr_stations, 
        seed, 
        policy = None
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.train_episodes = train_episodes
        self.test_episodes = test_episodes
        self.nr_stations = nr_stations
        self.seed = seed
        self.policy = policy


    def train(self, starting_loc):
        Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        
        rewards = []
        avg_rewards = []
        epsilons = []
        training_step = 0
        best_episode_reward = 0
        best_episode_segment = []
        # All the ACTUALIZED starting locations of the agent.
        actual_starting_locs = set()
        epsilon = self.initial_epsilon
        
        for episode in range(self.train_episodes):
            # Initialize starting location
            if starting_loc:
                if type(starting_loc[0]) == tuple:
                    loc = (random.randint(*starting_loc[0]), random.randint(*starting_loc[1]))
                else:
                    loc = starting_loc
            else:
                # Set the starting loc via e-greedy policy
                exp_exp_tradeoff = random.uniform(0, 1)
                # exploit
                if exp_exp_tradeoff > epsilon:
                    loc = tuple(self.env.city.vector_to_grid(np.unravel_index(Q.argmax(), Q.shape)[0]))
                # explore
                else:
                    loc = None

            if episode == 0:
                state, info = self.env.reset(seed=self.seed, loc=loc)
            else:
                state, info = self.env.reset(loc=loc)

            actual_starting_locs.add((state['location'][0], state['location'][1]))
            episode_reward = 0
            episode_step = 0
            while True:            
                state_index = self.env.city.grid_to_vector(state['location'][None, :]).item()

                # Exploration-exploitation trade-off 
                exp_exp_tradeoff = random.uniform(0, 1)

                # follow predetermined policy (set above)
                if self.policy:
                    action = self.policy[episode_step]
                # exploit
                elif exp_exp_tradeoff > epsilon:
                    action = np.argmax(Q[state_index, :] - 10000000 * (1-info['action_mask']))
                # explore
                else:
                    action = self.env.action_space.sample(mask=info['action_mask'])
                
                new_state, reward, done, _, info = self.env.step(action)

                # Here we sum the reward to create a single-objective policy optimization
                reward = reward.sum()

                # Update Q-Table
                new_state_gid = self.env.city.grid_to_vector(new_state['location'][None, :]).item()
                Q[state_index, action] = Q[state_index, action] + self.alpha * (reward + self.gamma * np.max(Q[new_state_gid, :]) - Q[state_index, action])
                episode_reward += reward

                training_step += 1
                episode_step += 1

                state = new_state

                if done:
                    break
            #Cutting down on exploration by reducing the epsilon
            epsilon = linearly_decaying_value(self.initial_epsilon, self.epsilon_decay_steps, episode, 0, self.final_epsilon)

            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                best_episode_segment = info['segments']
            
            # Adding the total reward and reduced epsilon values
            rewards.append(episode_reward)
            # Save the average reward over the last 10 episodes
            avg_rewards.append(np.average(rewards[-10:]))
            epsilons.append(epsilon)

            print(f'episode: {episode}, reward: {episode_reward} average rewards of last 10 episodes: {avg_rewards[-1]}')
        
        return Q, rewards, avg_rewards, epsilons, best_episode_reward, best_episode_segment, actual_starting_locs


def main(args):
    def make_env(gym_env):
        city = City(
            args.city_path,
            groups_file=args.groups_file,
            ignore_existing_lines=args.ignore_existing_lines
        )
        
        env = mo_gym.make(gym_env, 
                        city=city, 
                        constraints=MetroConstraints(city),
                        nr_stations=args.nr_stations)

        return env

    env = make_env(args.gym_env)
    
    agent = QLearningTNDP(
        env,
        alpha=args.alpha,
        gamma=args.gamma,
        initial_epsilon=args.initial_epsilon,
        final_epsilon=args.final_epsilon,
        epsilon_decay_steps=args.epsilon_decay_steps,
        train_episodes=args.train_episdes,
        test_episodes=args.test_episdes,
        nr_stations=args.nr_stations,
        seed=args.seed
    )
    agent.train(args.starting_loc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabular Q-learning for MO-TNDP")
    # Acceptable values: 'dilemma', 'margins', 'amsterdam'
    parser.add_argument('--env', default='dilemma', type=str)
    # For xian/amsterdam environment we have different groups files (different nr of objectives)
    parser.add_argument('--nr_groups', default=5, type=int)
    # Starting location of the agent
    parser.add_argument('--starting_loc_x', default=None, type=int)
    parser.add_argument('--starting_loc_y', default=None, type=int)
    parser.add_argument('--nr_stations', type=int)
    parser.add_argument('--alpha', default=0.4, type=float)
    parser.add_argument('--gamma', default=0.8, type=float)
    parser.add_argument('--initial_epsilon', default=1.0, type=float)
    parser.add_argument('--final_epsilon', default=0.0, type=float)
    parser.add_argument('--epsilon_decay_steps', default=400, type=float)
    parser.add_argument('--train_episdes', default=500, type=int)
    parser.add_argument('--test_episdes', default=1, type=int)
    parser.add_argument('--no_log', action='store_true', default=False)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    args.project_name = "TNDP-RL"

    # Some values are hardcoded for each environment (this is flexible, but we don't want to have to pass 100 arguments to the script)
    if args.env == 'dilemma':
        args.city_path = Path(f"./envs/mo-tndp/cities/dilemma_5x5")
        args.nr_stations = 9
        args.gym_env = 'motndp_dilemma-v0'
        args.groups_file = "groups.txt"
        args.ignore_existing_lines = True
        args.experiment_name = "Q-Learning-Dilemma"
    elif args.env == 'margins':
        args.city_path = Path(f"./envs/mo-tndp/cities/margins_5x5")
        args.nr_stations = 9
        args.gym_env = 'motndp_margins-v0'
        args.groups_file = f"groups.txt"
        args.ignore_existing_lines = True
        args.experiment_name = "Q-Learning-Margins"
    elif args.env == 'amsterdam':
        args.city_path = Path(f"./envs/mo-tndp/cities/amsterdam")
        args.nr_stations = 10
        args.gym_env = 'motndp_amsterdam-v0'
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.ignore_existing_lines = True
        args.experiment_name = "Q-Learning-Amsterdam"
    elif args.env == 'xian':
        args.city_path = Path(f"./envs/mo-tndp/cities/xian")
        args.nr_stations = 20
        args.gym_env = 'motndp_xian-v0'
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.ignore_existing_lines = True
        args.experiment_name = "Q-Learning-Xian"
        
    if args.starting_loc_x is not None and args.starting_loc_y is not None:
        args.starting_loc = (args.starting_loc_x, args.starting_loc_y)
    else:
        args.starting_loc = None

    main(args)