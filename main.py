import gym
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from nets import train_both, PolicyNet, ValueNet
from utils import History, Episode

seed = 2024
random.seed(seed)
np.random.seed(seed)

levels = 10
lr = 0.01
max_eps = 2
train_epochs = 5
max_timesteps = 400
state_scale = 1.
reward_scale = 20.
batch_size = 32

level_switch = 200
max_iterations = level_switch * levels

rp_range = 2

device = torch.device("mps")

def get_peturbations(env_name, seed):
  env = gym.make(env_name, render_mode="rgb_array")
  observation = env.reset()[0]
  random_perturbations = [
        np.random.normal(0, rp_range, observation.shape) for _ in range(levels)
    ]
  # make the first random perturbation zero
  random_perturbations[0] = np.zeros(observation.shape)
  return random_perturbations
def train(env_name, opt_choice, random_perturbations):

    # Create log txt files
    # trac_reward_log_file = f'logs/trac_reward_log_{env_name}_{seed}.txt'
    base_reward_log_file = f'logs/base_reward_log_{env_name}_{seed}.txt'

    # Setup env
    env = gym.make(env_name, render_mode="rgb_array")
    observation = env.reset()[0]
    n_actions = env.action_space.n
    feature_dim = observation.size

    tqdm_bar = tqdm(range(max_iterations), desc="Training", unit="iteration")

    value_model = ValueNet(in_dims=feature_dim).to(device)
    policy_model = PolicyNet(in_dims=feature_dim, n=n_actions).to(device)

    # trac_combined_optimizer = start_trac(log_file=f'logs/trac_{env_name}.text', Base=optim.Adam)(
    #     [
    #         {"params": policy_model.parameters(), "lr": lr},
    #         {"params": value_model.parameters(), "lr": lr},
    #     ]
    # )

    base_combined_optimizer = torch.optim.Adam(
        [
            {"params": policy_model.parameters(), "lr": lr},
            {"params": value_model.parameters(), "lr": lr},
        ]
    )
    # if opt_choice == "TRAC":
    #     combined_optimizer = trac_combined_optimizer
    #     reward_log_file = trac_reward_log_file
    #     print("USING TRAC.")
    if opt_choice == "base":
        combined_optimizer = base_combined_optimizer
        reward_log_file = base_reward_log_file
    history = History()
    level = 0
    for ite in tqdm_bar:
        # Switch perturbation level
        if ite % level_switch == 0:
            random_perturbation = random_perturbations[level]
            level += 1

        episodes_reward = []

        for _ in range(max_eps):
            observation = env.reset()[0]
            observation += random_perturbation
            episode = Episode()

            for timestep in range(max_timesteps):
                action, log_probability = policy_model.sample_action(observation / state_scale)
                value = value_model.state_value(observation / state_scale)

                new_observation, reward, done, _, _ = env.step(action)
                new_observation += random_perturbation

                episode.append(
                    obs=observation / state_scale,
                    action=action,
                    reward=reward,
                    value=value,
                    log_prob=log_probability,
                    reward_scale=reward_scale,
                )

                observation = new_observation

                if done:
                    episode.end_episode(last_value=0)
                    break

                if timestep == max_timesteps - 1:
                    value = value_model.state_value(observation / state_scale)
                    episode.end_episode(last_value=value)

            episodes_reward.append(reward_scale * np.sum(episode.rewards))
            history.add_episode(episode)

        mean_rewards = np.mean(episodes_reward)
        tqdm_bar.set_postfix(mean_rewards=mean_rewards)

        with open(reward_log_file, 'a') as f:
            f.write(str(mean_rewards) + '\n')
        history.build_dataset()
        data_loader = DataLoader(history, batch_size=batch_size, shuffle=True)
        train_both(policy_model, value_model, combined_optimizer, data_loader, train_epochs)
        history.free_mem()

import seaborn as sns
import matplotlib.pyplot as plt

def read_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

def plot(env_name, seed):
    # Read data from files
    # trac_data = read_data(f'logs/trac_reward_log_{env_name}_{seed}.txt')
    base_data = read_data(f'logs/base_reward_log_{env_name}_{seed}.txt')

    # Convert data to float
    # trac_data = [float(i) for i in trac_data]
    base_data = [float(i) for i in base_data]


    # Smooth trac and base data
    window = 5
    # trac_data = np.convolve(trac_data, np.ones(window) / window, mode='valid')
    base_data = np.convolve(base_data, np.ones(window) / window, mode='valid')

    # Create a plot with seaborn
    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 6))

    plt.plot(base_data, label='Adam PPO', color='#4a69bd')
    # plt.plot(trac_data, label='TRAC PPO', color='#b71540')

    plt.xlabel('Timesteps')
    plt.ylabel('Mean Episode Reward')
    plt.title(f'{env_name}', fontsize=24)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    peturbations = get_peturbations("Acrobot-v1", seed)
    print("`Online Peturbations are")
    print(peturbations)
    env = "Acrobot-v1"
    opt = "base"
    train(env, opt, peturbations)
    # plot(env, seed)