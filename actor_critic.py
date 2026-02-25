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


class Actor_Critic_Agent:
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
        self.I = 1
    
    def select_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32)
        log_probs = torch.log_softmax(self.policy_net(state_t) / self.Temperature, dim=0) #log_prob distribution of the actions
        probs = torch.exp(log_probs)
        action = torch.multinomial(probs, num_samples=1)
        return action.item(), log_probs[action] 
    
    def play(self):
        while self.num_episodes < self.max_episodes:
            state_t = self.state.copy()
            action, log_prob = self.select_action(self.state)
            self.state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.G += reward * self.discount
            self.discount *= self.gamma
            self.reward_list.append(reward)
            self.action_list.append(action)
            self.update_policy(state_t, reward, log_prob, done)
            if done:
                if self.num_episodes % 50 == 0:
                    print(f"Episode {self.num_episodes} finished with discounted reward {self.G:.2f}")
                self.discounted_reward_list.append(self.G)
                self.num_episodes += 1
                self.update_temperature()
                self.initialize()
            else:
                self.num_steps += 1
    
    def update_policy(self, state_t, reward, log_prob, done):
        state_t_tensor = torch.tensor(state_t, dtype=torch.float32)
        v_current = self.value_net(state_t_tensor).squeeze()

        if done:
            td_target = reward
        else:
            with torch.no_grad():
                v_next = self.value_net(torch.tensor(self.state, dtype=torch.float32)).squeeze()
            td_target = reward + self.gamma * v_next

        advantage = td_target - v_current
        value_loss = advantage ** 2
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        policy_loss = -advantage.detach() * self.I * log_prob
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.I *= self.gamma
        
    def update_temperature(self):
        self.Temperature = max(0.5, self.Temperature * 0.99)











if __name__ == "__main__":
    import os
    import numpy as np
    import random
    from plotting import plot_compare_smoothed_rewards

    num_seeds = 5
    os.makedirs("plots", exist_ok=True)

    lr_value_list = [1e-4, 5e-4, 1e-3, 2e-3]
    methods_rewards = []

    for lr_v in lr_value_list:
        seed_rewards = []
        for seed in range(num_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            env = gym.make('CartPole-v1')
            agent = Actor_Critic_Agent(env, EPS, lr_value=lr_v)
            agent.play()
            seed_rewards.append(agent.discounted_reward_list)
            print(f"αV={lr_v} seed {seed} done")
        methods_rewards.append(seed_rewards)

    plot_compare_smoothed_rewards(
        methods_rewards,
        labels=[f"αV={lr}" for lr in lr_value_list],
        window=50,
        title="Actor-Critic: Effect of Critic Learning Rate (CartPole-v1)",
        save_path="plots/actor_critic_lr_comparison.png",
    )