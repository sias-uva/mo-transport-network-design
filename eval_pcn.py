import datetime
import json
import os
from pathlib import Path
import mo_gymnasium as mo_gym
from morl_baselines.multi_policy.pcn.pcn_tndp import Transition
from motndp.city import City
from motndp.constraints import MetroConstraints
import numpy as np
import torch
import envs
import argparse

from morl_baselines.multi_policy.pcn.pcn_tndp import PCNTNDP

device = 'cpu'

def make_env(city_path, gym_env, nr_stations, groups_file, ignore_existing_lines):
    city = City(
        city_path, 
        groups_file=groups_file,
        ignore_existing_lines=ignore_existing_lines
    )
    
    env = mo_gym.make(gym_env, 
                    city=city, 
                    constraints=MetroConstraints(city),
                    nr_stations=nr_stations)

    return env

def greedy_action(model, obs, desired_return, desired_horizon, action_mask):
    probs = model(torch.tensor(np.expand_dims(obs, axis=0)).float().to(device),
                      torch.tensor(np.expand_dims(desired_return, axis=0)).float().to(device),
                      torch.tensor(np.expand_dims(desired_horizon, axis=0)).unsqueeze(1).float().to(device))
    # log_probs = log_probs.detach().cpu().numpy()[0]
    probs = probs.detach()
    # Apply the mask before log_softmax -- we add a large large number to the unmasked actions (Linear can return negative values)
    log_probs = torch.nn.functional.log_softmax(probs + action_mask * 10000, dim=-1)
    log_probs = log_probs.detach().cpu().numpy()[0]
    
    action = np.argmax(log_probs)
    
    return action


def run_episode(env, model, desired_return, desired_horizon, max_return, starting_loc=None):
    transitions = []
    state, info = env.reset(loc=starting_loc)
    states = [state['location']]
    obs = state['location_vector']
    done = False
    while not done:
        action = greedy_action(model, obs, desired_return, desired_horizon, info['action_mask'])
        n_state, reward, done, _, info = env.step(action)
        states.append(n_state['location'])
        n_obs = n_state['location_vector']

        transitions.append(Transition(
            observation=obs,
            action=action,
            action_mask=info['action_mask'],
            reward=np.float32(reward).copy(),
            next_observation=n_obs,
            terminal=done
        ))

        obs = n_obs
        # clip desired return, to return-upper-bound, 
        # to avoid negative returns giving impossible desired returns
        # reward = np.array((reward[1], reward[2]))
        desired_return = np.clip(desired_return-reward, None, max_return, dtype=np.float32)
        # clip desired horizon to avoid negative horizons
        desired_horizon = np.float32(max(desired_horizon-1, 1.))

    return transitions, states


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MO PCN - TNDP")
    parser.add_argument('--path', type=str, help='path of the model to evaluate')
    parser.add_argument('--checkpoint', type=int, default=None, help='integer index of the checkpoint to evaluate')
    parser.add_argument('--no_log', action='store_true', default=False)
    parser.add_argument('--interactive', action='store_true', default=False, help='interactive policy selection')

    args = parser.parse_args()

    model_dir = Path(args.path)
    # Load model and output file (contains the non-dominated set of solutions)
    with open(model_dir / 'output.txt') as f:
        output = f.read()
        output = json.loads(output)
    
    nd_front_r = np.array(output['best_front_r']) # rewards
    nd_front_h = np.array(output['best_front_h']) # horizons

    checkpoints = [str(p) for p in model_dir.glob('*.pt')]
    # Sort by creation date
    checkpoints.sort(key=lambda x: os.path.getmtime(x))
    if args.checkpoint is not None:
        model_to_eval = checkpoints[args.checkpoint]
    else:
        model_to_eval = checkpoints[-1]

    model = torch.load(model_to_eval)

    # Load the environment
    if output['env'] == 'motndp_dilemma-v0':
        city_path = Path(f"./envs/mo-tndp/cities/dilemma_5x5")
        nr_stations = 9
        nr_groups = len(output['best_front_r'][0])
        starting_loc = (output['starting_loc'][0], output['starting_loc'][1])
        groups_file = "groups.txt"
        ignore_existing_lines = True
        max_return=np.array([1, 1])
        starting_loc=(4, 0)
    elif output['env'] == 'motndp_amsterdam-v0':
        city_path = Path(f"./envs/mo-tndp/cities/amsterdam")
        nr_stations = 20
        nr_groups = len(output['best_front_r'][0])
        starting_loc = (output['starting_loc'][0], output['starting_loc'][1])
        groups_file = f"price_groups_{nr_groups}.txt"
        ignore_existing_lines = True
        max_return=np.array([1] * nr_groups)

    print(f'Will evaluate model {model_to_eval} , with groups file {groups_file}, starting location {starting_loc}')

    env = make_env(city_path, output['env'], nr_stations, groups_file, ignore_existing_lines)

    inp = -1
    while True:
        if args.interactive:
            print('solutions: ')
            for i, p in enumerate(nd_front_r):
                print(f'{i} : {p}')
            inp = input('-> ')
            inp = int(inp)
        else:
            inp += 1
            if inp >= len(nd_front_r):
                break

        desired_return = nd_front_r[inp]
        desired_horizon = nd_front_h[inp]

        # assume deterministic env, one run is enough
        transitions, states = run_episode(env, model, desired_return, desired_horizon, max_return, starting_loc)
        # compute return
        gamma = 1
        for i in reversed(range(len(transitions)-1)):
            transitions[i].reward += gamma * transitions[i+1].reward
        return_ = transitions[0].reward.flatten()

        print(f'ran model with desired-return: {desired_return.flatten()}, got {return_}')

        if not args.interactive:
            os.makedirs(model_dir / 'policies-executions', exist_ok=True)
            with open(model_dir / 'policies-executions' / f'policy_{inp}.txt', 'w') as f:
                f.write(', '.join(str(r) for r in return_))
