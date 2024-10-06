import gymnasium as gym
import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32

discrete_actions = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        ses, aes, res, s_es, dones = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_, done = transition
            ses.append(s)
            aes.append([a])
            res.append([r])
            s_es.append(s_)
            dones.append([done])

        return torch.tensor(ses, dtype=torch.float), torch.tensor(aes, dtype=torch.long), \
               torch.tensor(res, dtype=torch.float), torch.tensor(s_es, dtype=torch.float), \
               torch.tensor(dones, dtype=torch.float)

    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, len(discrete_actions))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x) 

    def sample_action(self, obs, epsilon):
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, len(discrete_actions) - 1)
        else:
            with torch.no_grad():
                q_values = self(obs)
                return q_values.argmax().item()

def train(q, q_target, memory, optimizer):
    for _ in range(10):
        s, a, r, s_, done = memory.sample(batch_size)
        q_out = q(s) 
        q_a = q_out.gather(1, a)  
        max_q_prime = q_target(s_).max(1)[0].unsqueeze(1) 
        target = r + gamma * max_q_prime * done
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    env = gym.make("Pendulum-v1")
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200))
        s, _ = env.reset()
        done = False

        while not done:
            action_idx = q.sample_action(torch.from_numpy(s).float(), epsilon)
            action = [discrete_actions[action_idx]] 
            s_, r, done, *_ = env.step(action)
            done = 0.0 if done else 1.0
            memory.put((s, action_idx, r/100.0, s_, done))
            s = s_

            score += r
            if done: break
        
        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)
        
        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode: {}, score: {:.1f}, n_buffer: {}, eps: {:.1f}%"\
                .format(n_epi, score/print_interval, memory.size(), epsilon * 100))
            
            score = 0.0
    env.close()

if __name__ == "__main__":
    main()