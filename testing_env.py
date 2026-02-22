#IMPORTS
#import gymnasium as gym
import minatar
#from minatar.gui import GUI
from minatar import Environment
#import minatar.gym
import random
#minatar.gym.register_envs()
env = Environment('breakout')
#env = gym.make('MinAtar/Breakout-v1')
#env.game.display_state(50)
num_actions = env.num_actions()
print(env.game_name(), env.n_channels)
env.display_state(50)
#gui = GUI(env.game_name(), env.n_channels)
G=0
terminated=False
while not terminated:
    action = random.randrange(num_actions)
    print("taking action", action)
    reward, terminated = env.act(action)
    s_prime = env.state()
    G+=reward
    env.display_state(500)

print(G)

# def func():
#     gui.display_state(env.state())
#     # one step of agent-environment interaction here
#     gui.update(50, func)
#     return 0

# gui.update(0, func)
# gui.run()