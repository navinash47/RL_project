import gymnasium as gym
import numpy as np
from Montecarlotreesearchcontinuous import MonteCarloTreeSearchContinuous
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

class Acrobot:
    def __init__(self):
        self.env = gym.make('Acrobot-v1', render_mode="rgb_array")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()[0]

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def check_goal_state(state,env):
    return -np.cos(state[0])-np.cos(state[1]+state[0])

def test_acrobot(mcts,iterations=500,depth=10):
    env = Acrobot()
    mcts = MonteCarloTreeSearchContinuous(env)

    state = env.reset()
    done = False
    total_reward = 0
    mcts.depth_limit = depth

    while not done:
        action = mcts.get_best_action(iterations=iterations)
        # print(check_goal_state(state,env))
        # print(f"current State: {state}, Action: {action}")
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        env.render()
    # print(check_goal_state(state,env))
    # print(f"Total reward: {total_reward}")
    env.close()
    return total_reward

def plotgraph_budget(budgets):
    budgets=[1,2,5,10,50,100,500,1000]
    avg_returns = []
    min_returns = [float('inf')]*len(budgets)
    max_returns = [float('-inf')]*len(budgets)
    for budget in tqdm(budgets,desc=f'Budget'):
        returns_for_specific_budget = []
        for i in range(10):
            total_reward = test_acrobot(budget)
            returns_for_specific_budget.append(total_reward)
        avg_returns.append(np.mean(returns_for_specific_budget))
        min_returns[budgets.index(budget)] = min(min_returns[budgets.index(budget)], np.min(returns_for_specific_budget))
        max_returns[budgets.index(budget)] = max(max_returns[budgets.index(budget)], np.max(returns_for_specific_budget))

    plt.plot(budgets, avg_returns, label='Average Return')
    plt.fill_between(budgets, min_returns, max_returns, alpha=0.2)
    plt.xscale('log')
    plt.xlabel('Budget')
    plt.ylabel('Return')
    plt.title('Monte Carlo Tree Search for Acrobot')
    plt.legend()
    plt.show()

def plotgraph_depth(depths):
    depths=[5,10,50,100,500]
    import matplotlib.pyplot as plt
    avg_returns_depth = []
    min_returns_depth = [float('inf')]*len(depths)
    max_returns_depth = [float('-inf')]*len(depths)
    env = Acrobot()
    mcts = MonteCarloTreeSearchContinuous(env)
    for depth in tqdm(depths,desc=f'Depth'):
        returns_for_specific_depth = []
        for i in range(10):
            total_reward = test_acrobot(mcts,iterations=100,depth=depth)
            returns_for_specific_depth.append(total_reward)
        avg_returns_depth.append(np.mean(returns_for_specific_depth))
        print(f"Average return for depth {depth}: {avg_returns_depth[-1]}")
        min_returns_depth[depths.index(depth)] = min(min_returns_depth[depths.index(depth)], np.min(returns_for_specific_depth))
        max_returns_depth[depths.index(depth)] = max(max_returns_depth[depths.index(depth)], np.max(returns_for_specific_depth))

    plt.plot(depths, avg_returns_depth, label='Average Return')
    plt.fill_between(depths, min_returns_depth, max_returns_depth, alpha=0.2)
    plt.xscale('log')
    plt.xlabel('Depth')
    plt.ylabel('Return')
    plt.title('Monte Carlo Tree Search for Acrobot')
    plt.legend()
    plt.show()



class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

def train_policy_network(env, mcts, policy_net, optimizer, num_iterations=10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = policy_net.to(device)

    for i in range(num_iterations):
        state = env.reset()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # Get action from MCTS and convert to one-hot probability distribution
        print(f"i:{i}:")
        action = mcts.get_best_action(iterations=100)
        mcts_probs = torch.zeros(1, 3)  # One-hot vector for 3 possible actions
        mcts_probs[0][action] = 1.0
        mcts_probs = mcts_probs.to(device)

        predicted_probs = policy_net(state_tensor)

        loss = nn.KLDivLoss(reduction='batchmean')(predicted_probs.log(), mcts_probs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.4f}")

def main():
    env = Acrobot()
    mcts = MonteCarloTreeSearchContinuous(env)

    input_size = 6  # Acrobot state size
    output_size = 3  # Number of actions

    policy_net = PolicyNetwork(input_size, output_size)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

    train_policy_network(env, mcts, policy_net, optimizer)

    # Save the trained model
    torch.save(policy_net.state_dict(), "acrobot_policy_net.pth")

    print("Training completed. Model saved as 'acrobot_policy_net.pth'")


if __name__ == "__main__":
    plotgraph_budget()
    plotgraph_depth()
    main()
    

    
