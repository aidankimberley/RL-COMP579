import gymnasium as gym
import minatar
#from minatar.gui import GUI
from minatar import Environment
import minatar.gym
import random
import torch
import matplotlib.pyplot as plt
N_INPUTS = 4
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

class Agent:
    def __init__(self, env, eps, lr, gamma):
        self.G = 0
        self.num_steps = 0
        self.num_episodes = 0
        self.max_episodes = 20000
        self.rewards_list = []
        self.is_terminated = 0
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
            action, action_value = self.select_action(self.env.state())
            reward, self.is_terminated = self.env.act(action)
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
        if self.num_steps<1:
            #use 1000 warmup steps to collect experience and explore before gradient update
            return
        else:

            #sequatial update of policy net no buffer for now
            with torch.no_grad():
                target_Q = self.target_net(torch.tensor(next_state.flatten(), dtype=torch.float32)).max().item() #max Q value of next state
                bellman_target = torch.tensor(reward + self.gamma * target_Q * (1-self.is_terminated), dtype=torch.float32)
            
            self.optimizer.zero_grad()
            loss = self.loss_fn(predicted_action_value, bellman_target)
            #print(f"requires_grad: {predicted_action_value.requires_grad}, grad_fn: {predicted_action_value.grad_fn}")
            loss.backward()
            # for name, param in self.policy_net.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name} | grad mean: {param.grad.abs().mean().item()}")
            #     else:
            #         print(f"WARNING: {name} has NO gradient!")
            # # Before optimizer.step()
            # print(f"Loss: {loss.item()}")

            # old_weights = {name: p.clone() for name, p in self.policy_net.named_parameters()}
            self.optimizer.step()
            # for name, p in self.policy_net.named_parameters():
            #     change = (p - old_weights[name]).abs().mean().item()
            #     print(f"{name} | mean weight change: {change}")


env = Environment('breakout')
# print(env.state())
# print(env.state().shape)

agent = Agent(env, EPS, LR, GAMMA)
print("num actions", agent.num_actions)
print("in channels", agent.in_channels)
agent.play()

plt.plot(agent.rewards_list)
plt.show()


#test the NN

# net = Net(N_OUTPUTS, N_INPUTS)
# state = torch.randn(N_INPUTS)
# action_distribution = net(state)
# action = torch.argmax(action_distribution)
# print(action)
        