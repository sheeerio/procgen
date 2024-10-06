import numpy as np
from torch.utils.data import Dataset
import torch

device = torch.device("mps")

def cumulative_sum(array, gamma=1.0):
    curr = 0
    cumulative_array = []

    for a in array[::-1]:
        curr = a + gamma * curr
        cumulative_array.append(curr)

    return cumulative_array[::-1]

class Episode:
    def __init__(self, gamma=0.99, lambd=0.95):
        self.obs = []
        self.actions = []
        self.advantages = []
        self.rewards = []
        self.rewards_to_go = []
        self.values = []
        self.log_probs = []
        self.gamma = gamma
        self.lambd = lambd

    def append(self, obs, action, reward, value, log_prob, reward_scale=20):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward / reward_scale)
        self.values.append(value)
        self.log_probs.append(log_prob)
    
    def end_episode(self, last_value):
        rewards = np.array(self.rewards + [last_value])
        values = np.array(self.values + [last_value])
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantages = cumulative_sum(deltas.tolist(), gamma = self.gamma * self.lambd)
        self.rewards_to_go = cumulative_sum(rewards.tolist(), gamma=self.gamma)[:-1]

def normalize_list(array):
    array = np.array(array)
    array = (array - np.mean(array)) / (np.std(array) + 1e-5)
    return array.tolist()

class History(Dataset):
    def __init__(self):
        self.episodes = []
        self.obs = []
        self.actions = []
        self.advantages = []
        self.rewards = []
        self.rewards_to_go = []
        self.log_probs = []

    def free_mem(self):
        del self.episodes[:]
        del self.obs[:]
        del self.actions[:]
        del self.advantages[:]
        del self.rewards[:]
        del self.rewards_to_go[:]
        del self.log_probs[:]
    
    def add_episode(self, episode):
        self.episodes.append(episode)
    
    def build_dataset(self):
        for episode in self.episodes:
            self.obs.extend(episode.obs)  # Flattening the list of lists by adding each observation
            self.actions.extend(episode.actions)  # Same for actions
            self.advantages.extend(episode.advantages)  # Same for advantages
            self.rewards.extend(episode.rewards)  # Same for rewards
            self.rewards_to_go.extend(episode.rewards_to_go)  # Same for rewards-to-go
            self.log_probs.extend(episode.log_probs)  # Same for log_probs

        assert(
            len(
                {
                    len(self.obs),
                    len(self.actions),
                    len(self.advantages),
                    len(self.rewards),
                    len(self.rewards_to_go),
                    len(self.log_probs),
                }
            ) == 1
        )
        self.advantages = normalize_list(self.advantages)
    
    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, index):
        return (
            torch.tensor(self.obs[index], dtype=torch.float32).to(device),
            torch.tensor(self.actions[index], dtype=torch.long).to(device),
            torch.tensor(self.advantages[index], dtype=torch.float32).to(device),
            torch.tensor(self.log_probs[index], dtype=torch.float32).to(device),
            torch.tensor(self.rewards_to_go[index], dtype=torch.float32).to(device),
        )