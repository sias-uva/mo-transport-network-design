# This is a simple demo of how to use Q-learning to solve the TNDP, using the gymnasium framework.
# Here we assume a single reward function, which is the sum of the rewards of all groups.
# Thus, one can say the utility function is the (equal weighted) sum of the utilities of all groups.
import datetime
from pathlib import Path
import random
from motndp.city import City
from motndp.constraints import MetroConstraints
import numpy as np
import gymnasium
import matplotlib.pyplot as plt
import envs

alpha = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # learning rate
gamma = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
epsilon = 1
max_epsilon = 1
min_epsilon = 0.00
e_decay = 0.05
train_episodes = 130

test_episodes = 1
nr_stations = 20
seed = 42
starting_loc = (11, 14)

# follow pre-determined policy
policy = None

def train(env: gymnasium.Env, city: City, alpha: float, gamma: float, epsilon: float, e_decay: float, train_episodes: int, seed: int, starting_loc: tuple):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    avg_rewards = []
    epsilons = []
    training_step = 0
    best_episode_reward = 0
    best_episode_segment = []
    for episode in range(train_episodes):
        state, info = env.reset(seed=seed, loc=starting_loc)
        episode_reward = 0
        episode_step = 0
        while True:            
            state_index = city.grid_to_vector(state['location'][None, :]).item()

            # Exploration-exploitation trade-off 
            exp_exp_tradeoff = random.uniform(0, 1)

            # follow predetermined policy (set above)
            if policy:
                action = policy[episode_step]
            # exploit
            elif exp_exp_tradeoff > epsilon:
                action = np.argmax(Q[state_index, :] * info['action_mask'])
            # explore
            else:
                action = env.action_space.sample(mask=info['action_mask'])
            
            new_state, reward, done, _, info = env.step(action)

            # Here we sum the reward to create a single-objective policy optimization
            reward = reward.sum()

            # Update Q-Table
            new_state_gid = city.grid_to_vector(new_state['location'][None, :]).item()
            Q[state_index, action] = Q[state_index, action] + alpha * (reward + gamma * np.max(Q[new_state_gid, :]) - Q[state_index, action])
            episode_reward += reward

            training_step += 1
            episode_step += 1

            state = new_state

            if done:
                break
        #Cutting down on exploration by reducing the epsilon 
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-e_decay * episode)

        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            best_episode_segment = info['segments']
        
        # Adding the total reward and reduced epsilon values
        rewards.append(episode_reward)
        # Save the average reward over the last 10 episodes
        avg_rewards.append(np.average(rewards[-10:]))
        epsilons.append(epsilon)

        print(f'episode: {episode}, reward: {episode_reward} average rewards of last 10 episodes: {avg_rewards[-1]}')

    return Q, rewards, avg_rewards, epsilons, best_episode_reward, best_episode_segment

if __name__ == '__main__':
    def gen_line_plot_grid(lines):
        """Generates a grid_x_max * grid_y_max grid where each grid is valued by the frequency it appears in the generated lines.
        Essentially creates a grid of the given line to plot later on.

        Args:
            line (list): list of generated lines of the model
            grid_x_max (int): nr of lines in the grid
            grid_y_mask (int): nr of columns in the grid
        """
        data = np.zeros((city.grid_x_size, city.grid_y_size))

        for line in lines:
            # line_g = city.vector_to_grid(line)

            for station in line:
                data[station[0], station[1]] += 1
        
        data = data/len(lines)

        return data
    
    city = City(
        Path(f"./envs/mo-tndp/cities/amsterdam"), 
        groups_file="price_groups_5.txt",
        ignore_existing_lines=True
    )
    
    env = gymnasium.make('motndp_amsterdam-v0', city = city, constraints=MetroConstraints(city), nr_stations = nr_stations)
    
    if (type(alpha) == list) & (type(gamma) == list):
        # Plot hypeparameter search results
        rewards = []
        avg_rewards = []
        labels = []
        for a in alpha:
            for g in gamma:
                Q, rwds, avg_rwrds, epsilons, best_episode_reward, best_episode_segment = train(env, city, a, g, epsilon, e_decay, train_episodes, seed, starting_loc)
                rewards.append(rwds)
                avg_rewards.append(avg_rwrds)
                labels.append(f'a={a}, g={g}')

        #Visualizing results and total reward over all episodes
        fig, ax = plt.subplots(figsize=(10, 5))
        # ax.plot(rewards, label='rewards', color='lightgray')
        for i, avg_rwrds in enumerate(avg_rewards):
            ax.plot(avg_rwrds, label=labels[i])
        ax.set_xlabel('Episode')
        ax.set_ylabel('Training total reward')
        ax.set_ylim(0, None)

        ax2 = ax.twinx()
        ax2.plot(epsilons, label='epsilon', color='orange')
        ax2.set_ylabel('Epsilon')
        ax2.set_ylim(0, 1)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=6)
        fig.savefig(Path(f"./results/hyperparameter_search_d{e_decay}_{datetime.datetime.today().strftime('%Y%m%d_%H_%M_%S.%f')}.png"))
    elif type(alpha) == float:
        Q, rewards, avg_rewards, epsilons, best_episode_reward, best_episode_segment = train(env, city, alpha, gamma, epsilon, e_decay, train_episodes, seed, starting_loc)

        #Visualizing results and total reward over all episodes
        x = range(train_episodes)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, rewards, label='rewards', color='lightgray')
        ax.plot(x, avg_rewards, label='average reward', color='blue')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Training total reward')
        ax.set_ylim(0, None)

        ax2 = ax.twinx()
        ax2.plot(x, epsilons, label='epsilon', color='orange')
        ax2.set_ylabel('Epsilon')
        ax2.set_ylim(0, 1)
        # ax.plot(x, epsilons, label='epsilon', color='orange')
        fig.suptitle('Average reward over all episodes in training')
        ax.set_title(f'Best episode reward: {np.round(best_episode_reward, 5)}, avg. reward last 10 episodes: {np.round(avg_rewards[-1], 5)}')
        fig.legend()
        fig.savefig(Path(f'./results/qlearning_ams_a{alpha}_g{gamma}_d{e_decay}_epis{train_episodes}.png'))

        # Print the Q table
        fig, ax = plt.subplots(figsize=(10, 5))
        Q_actions = Q.argmax(axis=1).reshape(city.grid_x_size, city.grid_y_size)
        Q_values = Q.max(axis=1).reshape(city.grid_x_size, city.grid_y_size)
        im = ax.imshow(Q_values, label='Q values', cmap='Blues', alpha=0.5)
        markers = ['\\uparrow', '\\nearrow', '\\rightarrow', '\\searrow', '\\downarrow', '\\swarrow', '\\leftarrow', '\\nwarrow']
        for a in range(8):
            cells = np.nonzero((Q_actions == a) & (Q_values > 0))
            ax.scatter(cells[1], cells[0], c='red', marker=rf"${markers[a]}$", s=10,)
        
        cbar = fig.colorbar(im)
        fig.suptitle('Q values and best actions')
        fig.savefig(Path(f'./results/qlearning_ams_qtable_a{alpha}_g{gamma}_d{e_decay}_epis{train_episodes}.png'))

        # Testing the agent
        total_rewards = 0
        generated_lines = []
        for episode in range(test_episodes):
            state, info = env.reset(seed=seed, loc=starting_loc)
            episode_reward = 0
            locations = []
            while True:
                state_index = city.grid_to_vector(state['location'][None, :]).item()
                locations.append(state['location'].tolist())
                action = np.argmax(Q[state_index,:] * info['action_mask'])
                new_state, reward, done, _, info = env.step(action)
                reward = reward.sum()
                episode_reward += reward      
                state = new_state    
                if done:
                    break
            total_rewards += episode_reward
            generated_lines.append(locations)

        plot_grid = gen_line_plot_grid(np.array(generated_lines))
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(plot_grid)
        fig.suptitle(f'Average Generated line \n reward: {episode_reward}')
        fig.savefig(Path(f'./results/qlearning_ams_line_a{alpha}_g{gamma}_d{e_decay}_epis{train_episodes}.png'))

        print('Line Segments: ', locations)
