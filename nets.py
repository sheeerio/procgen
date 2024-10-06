import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

class PolicyNet(nn.Module):
    def __init__(self, n=4, in_dims=128):
        super(PolicyNet, self).__init__()

        self.fc1 = torch.nn.Linear(in_dims, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.fc4 = torch.nn.Linear(128, n)
        self.l_relu = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.l_relu(self.fc1(x))
        x = self.l_relu(self.fc2(x))
        x = self.l_relu(self.fc3(x))
        x = self.fc4(x)
        x = F.softmax(x, dim=-1)
        return x

    def sample_action(self, state):

        if not state is torch.Tensor:
            state = torch.from_numpy(state).float().to(device)
        
        if len(state.size()) == 1:
            state = state.unsqueeze(0)
        
        y = self(state)
        dist = Categorical(y)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def best_action(self, state):

        if not state is torch.Tensor:
            state = torch.from_numpy(state).float().to(device)

        if len(state.size()) == 1:
            state = state.unsqueeze(0)

        y = self(state).squeeze()
        action = torch.argmax(y)

        return action.item()

    def eval_actions(self, states, actions):
        y = self(states)
        dist = Categorical(y)
        entropy = dist.entropy()
        log_probs = dist.log_prob(actions)

        return log_probs, entropy
    
class ValueNet(nn.Module):
    def __init__(self, in_dims=128):
        super(ValueNet, self).__init__()

        self.fc1 = nn.Linear(in_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)
        self.l_relu = nn.LeakyReLU(0.1)
    
    def forward(self, x):

        x = self.l_relu(self.fc1(x))
        x = self.l_relu(self.fc2(x))
        x = self.l_relu(self.fc3(x))
        x = self.fc4(x)

        return x.squeeze(1)

    def state_value(self, state):

        if not state is torch.Tensor:
            state = torch.from_numpy(state).float().to(device)
        
        if len(state.size()) == 1:
            state = state.unsqueeze(0)
        
        y = self(state)
        return y.item()

def ac_loss_clipped(new_log_probs, old_log_probs, advantages, epsilon_clip=0.2):
    probability_ratios = torch.exp(new_log_probs - old_log_probs)
    clipped_probability_ratios = torch.clamp(
        probability_ratios, 1-epsilon_clip, 1+epsilon_clip
    )
    surrogate_1 = probability_ratios * advantages
    surrogate_2 = clipped_probability_ratios * advantages

    return -torch.min(surrogate_1, surrogate_2)

def train_both(policy_model: PolicyNet, value_model: ValueNet, 
    combined_optimizer, data_loader, epochs=40, clip=0.2):
    c1 = 0.01 # entropy regularization coefficient
    c2 = 0.5 # value loss weight coefficient

    for epoch in range(epochs):
        policy_losses = []
        value_losses = []

        for obs, actions, advantages, log_probs, rewards_to_go in data_loader:
            obs = obs.float().to(device)
            actions = actions.long().to(device)
            advantages = advantages.float().to(device)
            old_log_probs = log_probs.float().to(device)
            rewards_to_go = rewards_to_go.float().to(device)

            combined_optimizer.zero_grad()

            new_log_probs, entropy = policy_model.eval_actions(obs, actions)
            policy_loss = (
                ac_loss_clipped(
                    new_log_probs,
                    old_log_probs,
                    advantages,
                    epsilon_clip=clip
                ).mean() - c1 * entropy.mean()
            )
            policy_losses.append(policy_loss.item())

            values = value_model(obs).squeeze(-1)
            # print(f"Shape of values: {values.shape}")
            # print(f"Shape of rewards_to_go: {rewards_to_go.shape}")
            value_loss = c2 * F.mse_loss(values, rewards_to_go)
            value_losses.append(value_loss.item())


            total_loss = policy_loss + value_loss

            total_loss.backward()
            combined_optimizer.step()