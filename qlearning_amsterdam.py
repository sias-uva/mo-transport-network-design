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

# alpha = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # learning rate
# gamma = [1]
alpha = 0.3
gamma = 1
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
e_decay = 0.01
train_episodes = 250

exploration_episodes = 500

test_episodes = 1
nr_stations = 20
seed = 42
# starting_loc = None
starting_loc = (11, 14)
# Starting location ranges to sample from, x and y coordinates are sampled separately
# starting_loc = ((8, 12), (11, 15))

# follow pre-determined policy
policy = None

def highlight_cells(cells, ax, **kwargs):
    """Highlights a cell in a grid plot. https://stackoverflow.com/questions/56654952/how-to-mark-cells-in-matplotlib-pyplot-imshow-drawing-cell-borders
    """
    for cell in cells:
        (y, x) = cell
        rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
        ax.add_patch(rect)
    return rect

def train(env: gymnasium.Env, city: City, alpha: float, gamma: float, epsilon: float, e_decay: float, train_episodes: int, seed: int, starting_loc: tuple = None):
    """Trains the agent using Q-learning.

    Args:
        env (gymnasium.Env): The environment to train the agent in.
        city (City): The city object.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        epsilon (float): The exploration-exploitation trade-off.
        e_decay (float): The decay rate of epsilon.
        train_episodes (int): The number of episodes to train the agent for.
        seed (int): The random seed for reproducibility.
        starting_loc (tuple): The starting location of the agent, if tuple of tuples then it samples from the range. If None, the starting location is random.

    Returns:
        Q (np.ndarray): The Q-table.
        rewards (list): The total reward for each episode.
        avg_rewards (list): The rolling 10-episode average reward.
        epsilons (list): The epsilon values for each episode.
        best_episode_reward (float): The total reward of the best episode.
        best_episode_segment (list): The line segments of the best episode.
    """
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    avg_rewards = []
    epsilons = []
    training_step = 0
    best_episode_reward = 0
    best_episode_segment = []
    # All the ACTUALIZED starting locations of the agent.
    actual_starting_locs = set()
    for episode in range(train_episodes):
        # Initialize starting location
        if starting_loc:
            if type(starting_loc[0]) == tuple:
                loc = (random.randint(*starting_loc[0]), random.randint(*starting_loc[1]))
            else:
                loc = starting_loc
        else:
            loc = None

        if episode == 0:
            state, info = env.reset(seed=seed, loc=loc)
        else:
            state, info = env.reset(loc=loc)

        actual_starting_locs.add((state['location'][0], state['location'][1]))
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

    return Q, rewards, avg_rewards, epsilons, best_episode_reward, best_episode_segment, actual_starting_locs

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
    # For figure naming
    param_string = f"a{alpha}_g{gamma}_d{e_decay}_epis{train_episodes}_{datetime.datetime.today().strftime('%Y%m%d_%H_%M_%S.%f')}"
    
    if (type(alpha) == list) & (type(gamma) == list):
        # Plot hypeparameter search results
        rewards = []
        avg_rewards = []
        labels = []
        for a in alpha:
            for g in gamma:
                Q, rwds, avg_rwrds, epsilons, best_episode_reward, best_episode_segment, actual_starting_locs = train(env, city, a, g, epsilon, e_decay, train_episodes, seed, starting_loc)
                rewards.append(rwds)
                avg_rewards.append(avg_rwrds)
                labels.append(f'a={a}, g={g}')

        #Visualizing results and total reward over all episodes
        fig, ax = plt.subplots(figsize=(15, 10))
        # ax.plot(rewards, label='rewards', color='lightgray')

        #### To better distinguish between the different lines, we use a color map and different line styles.
        # From https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib
        NUM_COLORS = 50
        LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted', (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5, 1, 5))]
        NUM_STYLES = len(LINE_STYLES)
        cm = plt.get_cmap('gist_rainbow')
        ####

        for i, avg_rwrds in enumerate(avg_rewards):
            ax.plot(avg_rwrds, label=labels[i], c=cm(i//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS), linestyle=LINE_STYLES[i%NUM_STYLES])
        ax.set_xlabel('Episode')
        ax.set_ylabel('Training total reward')
        ax.set_ylim(0, None)

        ax2 = ax.twinx()
        ax2.plot(epsilons, label='epsilon', color='orange')
        ax2.set_ylabel('Epsilon')
        ax2.set_ylim(0, 1)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=9)
        fig.savefig(Path(f"./results/hyperparameter_search_d{e_decay}_{datetime.datetime.today().strftime('%Y%m%d_%H_%M_%S.%f')}.png"))
    elif type(alpha) == float:
        Q, rewards, avg_rewards, epsilons, best_episode_reward, best_episode_segment, actual_starting_locs = train(env, city, alpha, gamma, epsilon, e_decay, train_episodes, seed, starting_loc)

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
        fig.savefig(Path(f'./results/qlearning_ams_{param_string}.png'))

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
        highlight_cells(actual_starting_locs, ax=ax, color='limegreen')
        fig.savefig(Path(f'./results/qlearning_ams_qtable_{param_string}.png'))

        # Testing the agent
        total_rewards = 0
        generated_lines = []
        # test_starting_loc = tuple(city.vector_to_grid(Q.sum(axis=1).argmax()))
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
        highlight_cells([starting_loc], ax=ax, color='limegreen')
        fig.suptitle(f'Average Generated line \n reward: {episode_reward}')
        fig.savefig(Path(f'./results/qlearning_ams_line_{param_string}.png'))

        print('Line Segments: ', locations)
