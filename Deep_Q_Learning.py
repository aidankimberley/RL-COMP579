#import gymnasium as gym
import minatar
#from minatar.gui import GUI
from minatar import Environment
#import minatar.gym
import random
import torch
import matplotlib.pyplot as plt
import numpy as np

N_INPUTS = 400
N_OUTPUTS = 4
LR = 1e-3
GAMMA = 0.99
EPS = 0.05


class Net(torch.nn.Module):
    def __init__(self, n_actions, in_channels):
        super(Net, self).__init__()
        n_layer1 = 128
        n_layer2 = 128
        self.hidden1 = torch.nn.Linear(in_channels, n_layer1)
        self.hidden2 = torch.nn.Linear(n_layer1, n_layer2)
        self.output = torch.nn.Linear(n_layer2, n_actions)

        # Use default PyTorch initialization (Kaiming) - good for ReLU
        # Or explicitly use Xavier:
        # torch.nn.init.xavier_uniform_(self.hidden1.weight)
        # torch.nn.init.xavier_uniform_(self.hidden2.weight)
        # torch.nn.init.xavier_uniform_(self.output.weight)
        pass  # Default init is fine

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity, batch_size=64, alpha=0.6, beta=0.4):
        self.buffer_capacity = capacity
        self.batch_size = batch_size
        self.transitions = []
        self.priorities = []
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0

    def push(self, transition, steps):
        if len(self.transitions) < self.buffer_capacity:
            self.transitions.append(transition)
            self.priorities.append(self.max_priority)
        else:
            idx = steps % self.buffer_capacity
            self.transitions[idx] = transition
            self.priorities[idx] = self.max_priority

    def uniform_buffer_sample(self):
        indices = random.sample(range(len(self.transitions)), self.batch_size)
        return self._build_batch(indices), indices, None

    def priority_buffer_sample(self):
        probs = np.array(self.priorities, dtype=np.float64) ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.transitions), size=self.batch_size, p=probs, replace=False)
        weights = (len(self.transitions) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        importance_weights = torch.tensor(weights, dtype=torch.float32)
        return self._build_batch(indices), indices, importance_weights

    def update_priorities(self, indices, td_errors):
        for idx, td_err in zip(indices, td_errors):
            self.priorities[idx] = abs(td_err) + 1e-6
            self.max_priority = max(self.max_priority, self.priorities[idx])

    def update_beta(self, steps):
        self.beta = min(1.0, 0.4 + 0.6 * (steps / self.buffer_capacity))

    def _build_batch(self, indices):
        batch = [self.transitions[i] for i in indices]
        states = torch.from_numpy(np.array([t[0] for t in batch])).float()
        actions = torch.tensor([t[1] for t in batch], dtype=torch.long)
        rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32)
        next_states = torch.from_numpy(np.array([t[3] for t in batch])).float()
        is_terminated = torch.tensor([t[4] for t in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, is_terminated


class Agent:
    def __init__(self, env, eps, lr, gamma, batch_style='priority_batch'):
        self.G = 0
        self.num_steps = 0
        self.num_episodes = 0
        self.max_episodes = 3000
        self.rewards_list = []
        self.is_terminated = 0
        self.batch_size = 64
        self.buffer_capacity = 50000
        self.in_channels = env.n_channels * 10 * 10
        self.num_actions = env.num_actions() 
        self.env = env
        self.eps = eps
        self.lr = lr
        self.gamma = gamma
        self.policy_net = Net(self.num_actions, self.in_channels)
        self.target_net = Net(self.num_actions, self.in_channels)
        self.loss_fn = torch.nn.MSELoss() #squared L2 loss of predicted and bellman target
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr)
        self.batch_style = batch_style #sequential, uniform_batch, priority_batch
        self.transition_history = []
        self.replay_buffer = ReplayBuffer(self.buffer_capacity)

    def initialize(self):
        self.env.reset()
        self.is_terminated = 0
        self.G = 0


    def select_action(self, state):
        #must input a tensor not numpy array
        action_distribution = self.policy_net(torch.tensor(state.flatten(), dtype=torch.float32))
        if random.random() < self.eps:
            action = random.randint(0, self.num_actions - 1)
            action_value = action_distribution[action]
        else: 
            action = action_distribution.argmax().item()
            action_value = action_distribution[action]
        #returns int
        return action, action_value

    def play(self):
        while self.num_episodes < self.max_episodes:
            state = self.env.state()
            action, action_value = self.select_action(state)
            reward, self.is_terminated = self.env.act(action)
            self.replay_buffer.push((state, action, reward, self.env.state(), self.is_terminated), self.num_steps)
            self.update_neural_net(action_value, reward, self.env.state())
            self.G += reward
            if self.is_terminated:
                self.rewards_list.append(self.G)
                if self.num_episodes % 10 == 0:
                    print(f"Episode {self.num_episodes} over! Total reward: {self.G}")
                self.num_episodes += 1
                self.initialize()
            else:
                self.num_steps += 1
    
    def update_target_net(self):
        #set target = policy net weights
        self.target_net.load_state_dict(self.policy_net.state_dict())

    



    def update_neural_net(self, predicted_action_value, reward, next_state):
        if self.num_steps%1000==0:
            print("updating target net")
            self.update_target_net()
        if self.num_steps<1000:
            #use 1000 warmup steps to collect experience and explore before gradient update
            return
        else:
            if self.batch_style == 'sequential':
                #sequatial update of policy net no buffer for now
                with torch.no_grad():
                    target_Q = self.target_net(torch.tensor(next_state.flatten(start_dim=1), dtype=torch.float32)).max().item() #max Q value of next state
                    bellman_target = torch.tensor(reward + self.gamma * target_Q * (1-self.is_terminated), dtype=torch.float32)
                
                self.optimizer.zero_grad()
                loss = self.loss_fn(predicted_action_value, bellman_target)
                #print(f"requires_grad: {predicted_action_value.requires_grad}, grad_fn: {predicted_action_value.grad_fn}")
                loss.backward()
                self.optimizer.step()
            if self.batch_style == 'uniform_batch':
                (states, actions, rewards, next_states, is_terminated), _, _ = self.replay_buffer.uniform_buffer_sample()
                predicted_Qs = self.policy_net(states.flatten(start_dim=1)).gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    target_Qs = self.target_net(next_states.flatten(start_dim=1)).max(dim=1)[0]
                    bellman_targets = rewards + self.gamma * target_Qs * (1 - is_terminated)
                self.optimizer.zero_grad()
                loss = self.loss_fn(predicted_Qs, bellman_targets)
                loss.backward()
                self.optimizer.step()
            if self.batch_style == 'priority_batch':
                (states, actions, rewards, next_states, is_terminated), indices, importance_weights = self.replay_buffer.priority_buffer_sample()
                predicted_Qs = self.policy_net(states.flatten(start_dim=1)).gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    target_Qs = self.target_net(next_states.flatten(start_dim=1)).max(dim=1)[0]
                    bellman_targets = rewards + self.gamma * target_Qs * (1 - is_terminated)
                td_errors = (bellman_targets - predicted_Qs).detach()
                self.replay_buffer.update_priorities(indices, td_errors.numpy())
                self.replay_buffer.update_beta(self.num_steps)
                self.optimizer.zero_grad()
                loss = (importance_weights * (predicted_Qs - bellman_targets) ** 2).mean()
                loss.backward()
                self.optimizer.step()


env = Environment('breakout')
# print(env.state())
# print(env.state().shape)

agent = Agent(env, EPS, LR, GAMMA, batch_style='priority_batch')
# print("num actions", agent.num_actions)
# print("in channels", agent.in_channels)
agent.play()

# plt.plot(agent.rewards_list)
# plt.show()


#test the NN

# net = Net(N_OUTPUTS, N_INPUTS)
# state = torch.randn(N_INPUTS)
# action_distribution = net(state)
# action = torch.argmax(action_distribution)
# # print(action)
# random_tensor = torch.randn(64, 10, 10, 4)
# print(random_tensor.shape)
# net = Net(N_OUTPUTS, N_INPUTS)
# action_distribution = net(random_tensor.flatten(start_dim=1))
# print(action_distribution.shape)