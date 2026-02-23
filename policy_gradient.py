#implement REINFORCE base & with baseline
#implement one step actor critic
from typing import Any
from Deep_Q_Learning import Net
import gymnasium as gym
import torch
import matplotlib.pyplot as plt

N_INPUTS = 4
N_OUTPUTS = 2
LR = 1e-3
GAMMA = 0.99
EPS = 0.05

class REINFORCE_Agent:
    def __init__(self, env, eps, lr, gamma):
        self.env = env
        self.eps = eps
        self.state = None
        self.lr = lr
        self.gamma = gamma
        self.max_episodes = 1000
        self.num_episodes = 0
        self.G = 0
        self.num_steps = 0
        self.policy_net = Net(N_OUTPUTS, N_INPUTS)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr)
        self.Temperature = 10.0
        self.reward_list = []
        self.log_prob_list = []
        self.state_list = []
        self.action_list = []
        self.returns_list = []
        self.loss = 0
        self.initialize()
        self.discount = 1
        self.discounted_reward_list = []

    def initialize(self):
        self.state, _ = self.env.reset()
        self.G = 0
        self.returns_list = []
        self.reward_list = []
        self.log_prob_list = []
        self.state_list = []
        self.action_list = []
        self.loss = 0
        self.num_steps = 0
        self.discount = 1
    
    def select_action(self, state):
        logits = self.policy_net(torch.tensor(state, dtype=torch.float32)) / self.Temperature
        boltzmann_action_distribution = torch.softmax(logits, dim=0)
        action = torch.multinomial(boltzmann_action_distribution, num_samples=1)
        log_prob = torch.log(boltzmann_action_distribution[action])
        return action.item(), log_prob
    
    def play(self):
        while self.num_episodes < self.max_episodes:
            action, log_prob = self.select_action(self.state)
            self.state, reward, terminated, _, _ = self.env.step(action)
            self.G += reward * self.discount
            self.discount *= self.gamma
            self.reward_list.append(reward)
            self.action_list.append(action)
            self.state_list.append(self.state)
            self.log_prob_list.append(log_prob)
            if terminated:
                if self.num_episodes % 50 == 0:
                    print(f"Episode {self.num_episodes} finished with discounted reward {self.G:.2f}")
                self.discounted_reward_list.append(self.G)
                self.num_episodes += 1
                self.update_policy()
                self.update_temperature()
                self.initialize()
            else:
                self.num_steps += 1
    
    def update_policy(self):
        G = 0
        for reward in reversed(self.reward_list):
            G = self.gamma * G + reward
            self.returns_list.insert(0, G)
        self.loss = 0
        for returns, log_prob in zip(self.returns_list, self.log_prob_list):
            self.loss -= returns * log_prob
        self.loss = self.loss / len(self.returns_list)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        
    def update_temperature(self):
        self.Temperature = max(0.5, self.Temperature * 0.99)







class REINFORCE_Agent_with_Baseline:
    def __init__(self, env, eps, lr_policy = 1e-4, lr_value = 1e-3, gamma = 0.99):
        self.env = env
        self.eps = eps
        self.state = None
        self.lr = lr_policy
        self.lr_value = lr_value
        self.gamma = gamma
        self.max_episodes = 1000
        self.num_episodes = 0
        self.G = 0
        self.num_steps = 0
        self.policy_net = Net(N_OUTPUTS, N_INPUTS)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr_policy)
        self.Temperature = 10.0
        self.reward_list = []
        self.log_prob_list = []
        self.state_list = []
        self.action_list = []
        self.returns_list = []
        self.policy_loss = 0
        self.value_loss = 0
        self.initialize()
        self.discount = 1
        self.discounted_reward_list = []
        self.value_net = Net(1, N_INPUTS)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr_value)
        self.state_value_list = []

    def initialize(self):
        self.state, _ = self.env.reset()
        self.G = 0
        self.returns_list = []
        self.reward_list = []
        self.log_prob_list = []
        self.state_list = []
        self.action_list = []
        self.state_value_list = []
        self.policy_loss = 0
        self.value_loss = 0
        self.num_steps = 0
        self.discount = 1
    
    def select_action(self, state):
        logits = self.policy_net(torch.tensor(state, dtype=torch.float32)) / self.Temperature
        boltzmann_action_distribution = torch.softmax(logits, dim=0)
        action = torch.multinomial(boltzmann_action_distribution, num_samples=1)
        log_prob = torch.log(boltzmann_action_distribution[action])
        return action.item(), log_prob
    
    def play(self):
        while self.num_episodes < self.max_episodes:
            action, log_prob = self.select_action(self.state)
            self.state, reward, terminated, _, _ = self.env.step(action)
            self.G += reward * self.discount
            self.discount *= self.gamma
            self.reward_list.append(reward)
            self.action_list.append(action)
            self.state_list.append(self.state)
            self.log_prob_list.append(log_prob)
            if terminated:
                if self.num_episodes % 50 == 0:
                    print(f"Episode {self.num_episodes} finished with discounted reward {self.G:.2f}")
                self.discounted_reward_list.append(self.G)
                self.num_episodes += 1
                self.update_policy()
                self.update_temperature()
                self.initialize()
            else:
                self.num_steps += 1
    
    def update_policy(self):
        G = 0
        for reward in reversed(self.reward_list):
            G = self.gamma * G + reward
            self.returns_list.insert(0, G)
        policy_loss = 0
        value_loss = 0
        n = len(self.returns_list)
        for state, returns, log_prob in zip(self.state_list, self.returns_list, self.log_prob_list):
            value = self.value_net(torch.tensor(state, dtype=torch.float32)).squeeze()
            advantage = returns - value.detach().item()
            policy_loss -= advantage * log_prob
            value_loss += (value - returns) ** 2
        self.policy_optimizer.zero_grad()
        (policy_loss / n).backward()
        self.policy_optimizer.step()
        self.value_optimizer.zero_grad()
        (value_loss / n).backward()
        self.value_optimizer.step()
        
    def update_temperature(self):
        self.Temperature = max(0.5, self.Temperature * 0.99)









#VERIFY CODE
Temperature = 10.0
env = gym.make('CartPole-v1')
state, info = env.reset()

agent = REINFORCE_Agent_with_Baseline(env, EPS)
agent.play()
print(agent.discounted_reward_list)
plt.plot(agent.discounted_reward_list)
plt.show()

# print(state.shape)
# print(env.action_space.n)

# network = Net(N_OUTPUTS, N_INPUTS)
# logits = network(torch.tensor(state, dtype=torch.float32)) / Temperature
# boltzmann_action_distribution = torch.softmax(logits,dim=0)
# action = torch.multinomial(boltzmann_action_distribution, num_samples=1)
# print(action.item())