import numpy as np
import gymnasium as gym
from Montecarlotreesearchdiscrete import MonteCarloSearchTree
import matplotlib.pyplot as plt
from tqdm import tqdm

class CliffWalking:
    def __init__(self):
        # Initialize gym environment
        self.env = gym.make('CliffWalking-v0')
        
        # Grid dimensions from gym
        self.states = self.env.observation_space.n
        self.actions = list(range(self.env.action_space.n))  # Convert to list to make iterable
        self.current_state = self.env.reset()[0]
        
        # Define states
        self.cliff = [(3,j) for j in range(1,11)]  # Cliff positions in bottom row
        self.goal = (3,11)  # Bottom right corner
        self.current_state = self.env.reset()[0]

    def get_next_state(self, state, action):
        # Save current state
        original_state = self.env.unwrapped.s
        
        # Convert state coordinates to single integer if needed
        if isinstance(state, tuple):
            state = state[0] * 12 + state[1]  # Convert (row, col) to single integer
        
        # Set environment to desired state and reset if needed
        self.env.unwrapped.s = state
        if not hasattr(self.env, '_has_reset') or not self.env._has_reset:
            self.env.reset()
            self.env.unwrapped.s = state
            
        # Get next state
        next_state = self.env.step(action)[0]
        
        # Restore original state
        self.env.unwrapped.s = original_state
        return next_state

    def step(self,action):
        return self.env.step(action)[0], self.env.step(action)[1], self.env.step(action)[2]

def plot_budget():
    budgets=[1,2,5,10,50,100,500,1000]
    avg_returns = []
    min_returns = [float('inf')]*len(budgets)
    max_returns = [float('-inf')]*len(budgets)
    for budget in tqdm(budgets,desc=f'Budget'):
        returns_for_specific_budget = []
        for i in range(10):
            total_reward = test_cliff_walking(budget)
            returns_for_specific_budget.append(total_reward)
        avg_returns.append(np.mean(returns_for_specific_budget))
        min_returns[budgets.index(budget)] = min(min_returns[budgets.index(budget)], np.min(returns_for_specific_budget))
        max_returns[budgets.index(budget)] = max(max_returns[budgets.index(budget)], np.max(returns_for_specific_budget))

def plot_depth(mcts,env,iterations=10000,depth=200):
    
    mcts = MonteCarloSearchTree(env)
    # Convert initial state (0,0) to integer representation
    mcts.root.state = 0  # (0 * 12 + 0)
    values = []
    mcts.depth_limit = depth
    for _ in tqdm(range(5)):
        p,value = mcts.get_best_action(iterations)
        values.append(value)
    return min(values), max(values), sum(values)/len(values)

    depths=[1,2,5,10,50,100]
    avg_returns_depth = []
    min_returns_depth = [float('inf')] * len(depths) 
    max_returns_depth = [float('-inf')] * len(depths)

    for depth in tqdm(depths,desc=f'Depth'):
        min_return, max_return, avg_return = plot_depth(mcts,env,iterations,depth)
        min_returns_depth[depths.index(depth)] = min_return
        max_returns_depth[depths.index(depth)] = max_return
        avg_returns_depth.append(avg_return)
        

    plt.plot(depths, avg_returns_depth, label='Average Return')
    plt.plot(depths, min_returns_depth, label='Min Return')
    plt.plot(depths, max_returns_depth, label='Max Return')
    plt.fill_between(depths, min_returns_depth, max_returns_depth, alpha=0.2)
    # plt.xscale('log')
    plt.xlabel('Depth')
    plt.ylabel('Return')
    plt.title('Monte Carlo Tree Search for Cliff Walking')
    plt.legend()
    plt.show()

# Move the function definition outside the multiprocessing code and wrap everything in a main function
def run_single_iteration(state, env, mcts):
    print(f'Processing state: {state}',end=':')
    action_count = [0,0,0,0]  # Up, Right, Down, Left
    if state != 47:  # If not goal state (3,11)
        env.current_state = state
        mcts.root = Node(state)
        probs_pi_state, optimal_value = mcts.get_best_action(1000, min_visits=2000)
        optimal_action = np.argmax(probs_pi_state)
        action_count[optimal_action] += 1
        print(f'op: {optimal_action}')
    return probs_pi_state, optimal_value, action_count

def main():
    env = CliffWalking()
    mcts = MonteCarloSearchTree(env)
    mcts.depth_limit = 50
    probs_pi = {}
    pi = {}
    with tqdm(total=4*12, desc='Processing states') as pbar:
        for i in range(4):
            for j in range(12):
                state = i * 12 + j
                # Replace multiprocessing with sequential processing
                results = [run_single_iteration(state, env, mcts) for _ in range(20)]
                
                # Initialize arrays to store sums
                probs_sum = np.zeros_like(results[0][0])
                value_sum = 0
                action_count = [0, 0, 0, 0]
                
                # Sum up results from all iterations
                for probs, value, counts in results:
                    probs_sum += probs
                    value_sum += value
                    for k in range(4):
                        action_count[k] += counts[k]
                        
                # Calculate averages
                probs_pi[state] = probs_sum / len(results)
                probs_pi[state] = np.exp(probs_sum) / np.sum(np.exp(probs_sum))
                optimal_value = value_sum / len(results)
                if state != 47:  # If not goal state
                    pi[state] = np.argmax(probs_pi[state])
                    print(f'state: {(i,j)},action_count: {action_count}, pi[{state}]: {pi[state]}')
                # if state != 47:  # If not goal state
                #     pi[state] = action_count.index(max(action_count))
                #     print(f'state: {(i,j)}, action_count: {action_count}, pi[{state}]: {pi[state]}')
            pbar.update(1)
    
    return probs_pi, pi


if __name__ == "__main__":
    env = CliffWalking()
    mcts = MonteCarloSearchTree(env)
    plot_depth(mcts,env)
    probs_pi, pi = main()