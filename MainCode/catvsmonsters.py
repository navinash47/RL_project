import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from Montecarlotreesearchcontinuous import MonteCarloSearchTree,Node

class catVsMonsters:
    def __init__(self):
        # Grid dimensions
        self.rows = 5
        self.cols = 5
        self.states = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        
        # Define states
        self.furniture = [(2, 1), (2, 2), (2, 3), (3, 2)]  # Obstacles
        self.monsters = [(4, 1), (0, 3)]  # Danger state
        self.food = [(4, 4)]  # Goal state
        self.goal = (4, 4)  # Explicit goal state
        
        # Define actions
        self.actions = ['AU', 'AD', 'AL', 'AR']  # Match GridWorld actions
        self.action_effects = {
            'AU': (-1, 0),
            'AD': (1, 0),
            'AL': (0, -1), 
            'AR': (0, 1)
        }
        
        # Action probabilities
        self.p_intended = 0.7   # 70% intended direction
        self.p_right = 0.12     # 12% for right deviation
        self.p_left = 0.12      # 12% for left deviation
        self.p_stay = 0.06      # 6% stay in place
        
        # Initialize current state
        self.current_state = (0, 0)
        
    def get_next_state(self, state, action):
        """Get next state based on current state and action"""
        if action not in self.actions:
            raise ValueError("Invalid action")
            
        # Get movement vector
        dr, dc = self.action_effects[action]
        new_r = state[0] + dr
        new_c = state[1] + dc
        
        # Check if move is valid
        if (new_r >= 0 and new_r < self.rows and 
            new_c >= 0 and new_c < self.cols and
            (new_r, new_c) not in self.furniture):
            return (new_r, new_c)
        return state
    
    def step(self, action):
        """Take a step in the environment"""
        if action not in self.actions:
            raise ValueError("Invalid action")
            
        # Determine actual movement based on probabilities
        p = np.random.random()
        
        if p < self.p_stay:  # 6% chance to stay
            actual_action = None
        elif p < self.p_intended + self.p_stay:  # 70% chance for intended action
            actual_action = action
        elif p < self.p_intended + self.p_stay + self.p_left:  # 12% chance for left deviation
            if action == 'AU':
                actual_action = 'AL'
            elif action == 'AD':
                actual_action = 'AR'
            else:
                actual_action = 'AU'
        else:  # 12% chance for right deviation
            if action == 'AU':
                actual_action = 'AR'
            elif action == 'AD':
                actual_action = 'AL'
            else:
                actual_action = 'AD'
        
        # Update state
        if actual_action:
            self.current_state = self.get_next_state(self.current_state, actual_action)
        
        # Calculate reward
        if self.current_state == self.goal:
            reward = 10
        elif self.current_state in self.monsters:
            reward = -8
        else:
            reward = -0.05  # Small step penalty
        
        # Check if episode is done
        done = self.current_state == self.goal
        
        return self.current_state, reward, done
            
    def reset(self):
        """Reset environment to initial state"""
        self.current_state = (0, 0)
        return self.current_state

    def get_q_value_for_vi(self, state, action, course="intended", gamma=0.9, V=None):
        """Get the q value for the given state and action"""
        if course == "intended":
            # Calculate reward based on next state
            next_state = self.get_next_state(state, action)
            reward = self._get_reward(next_state)
            return reward + gamma * V[next_state]
        elif course == "right":
            next_state = self.get_next_state(state, self.right_of(action))
            reward = self._get_reward(next_state)
            return reward + gamma * V[next_state]
        elif course == "left":
            next_state = self.get_next_state(state, self.left_of(action))
            reward = self._get_reward(next_state)
            return reward + gamma * V[next_state]
        elif course == "stay":
            # Calculate reward for staying in the same state
            reward = self._get_reward(state)
            return reward + gamma * V[state]

    def _get_reward(self, state):
        """Helper method to calculate reward for a given state"""
        if state == self.goal:
            return 10
        elif state in self.monsters:
            return -8
        else:
            return -0.05

    def right_of(self, action):
        """Get the right of the given action"""
        if action == 'AU':
            return 'AR'
        elif action == 'AD':
            return 'AL'
        elif action == 'AL':
            return 'AU'
        elif action == 'AR':
            return 'AD'

    def left_of(self, action):
        """Get the left of the given action"""
        if action == 'AU':
            return 'AL'
        elif action == 'AD':
            return 'AR'
        elif action == 'AL':
            return 'AD'
        elif action == 'AR':
            return 'AU'

def print_policy(policy):
    for i in range(5):
        for j in range(5):
            state = (i, j)
            if state == env_cat.goal:
                print("G", end=" ")
                continue
            elif state in env_cat.furniture:
                print("F", end=" ")
                continue
            action = policy[state]
            if action == 0:
                print("↑", end=" ")
            elif action == 1:
                print("↓", end=" ")
            elif action == 2:
                print("←", end=" ")
            elif action == 3:
                print("→", end=" ")
            else:
                print(" ", end=" ")
        print()

def plot_graph_for_budget(env,min_visits=1000):
    mcts = MonteCarloSearchTree(env)
    mins={}
    maxs={}
    avgs={}
    budget=[500,1000,10000,100000]
    probs_pi_dict={}
    # time_slots=[1,5,10,50,100]
    for b in budget:
        min_value = float('inf')
        max_value = float('-inf')
        avg_value = 0
        avg_iteration = 0
        avg_pi=None
        for i in tqdm(range(5)):
            env.reset()
            mcts.root = Node((0,0))
            probs_pi,optimal_value = mcts.get_best_action(b,min_visits=min_visits)
            optimal_action = np.argmax(probs_pi)
            print(f'budget: {b}, optimal_value: {optimal_value}, optimal_action: {optimal_action}')
            min_value = min(min_value, optimal_value)
            max_value = max(max_value, optimal_value)
            avg_value += optimal_value
            if avg_pi is None:
                avg_pi = probs_pi
            else:
                avg_pi += probs_pi
        avg_pi /= 5
        probs_pi_dict[b] = avg_pi/5
        avg_value /= 5

        mins[b] = min_value
        maxs[b] = max_value
        avgs[b] = avg_value
    
    return mins,maxs,avgs,budget

def plot_budget():
    cat_dynamics = catVsMonsters()
    # cat_dynamics.current_state = (0,4)
    mins_cat,maxs_cat, avgs_cat, iterations_cat = plot_graph_for_test(cat_dynamics,min_visits=1)

    y_values =list(avgs_grid.values())
    x_values = [500,1000,10000,100000]
    y_values = [x for x in y_values]
    plt.plot(x_values,y_values,label='avg', marker='o')
    plt.xlabel('iterations or budget')
    plt.ylabel('Average Return')
    plt.title('Grid World')
    min_values_grid = [x for x in list(mins_grid.values())]
    max_values_grid = [x for x in list(maxs_grid.values())]  
    plt.fill_between(x_values,min_values_grid,max_values_grid,alpha=0.5)
    plt.legend()
    plt.xscale('log')
    plt.grid(True)
    plt.show()

# check for different depth limits

def plot_graph_for_test_depth(env,min_visits=1000):
    mcts = MonteCarloSearchTree(env)
    mins={}
    maxs={}
    avgs={}
    depths=[5,10,20,40,100,200]
    probs_pi_dict={}
    b=10000
    # time_slots=[1,5,10,50,100]
    for depth in depths:
        min_value = float('inf')
        max_value = float('-inf')
        avg_value = 0
        avg_iteration = 0
        avg_pi=None
        for i in tqdm(range(5)):
            env.reset()
            mcts.root = Node((0,0))
            probs_pi,optimal_value = mcts.get_best_action(b,min_visits=min_visits)
            optimal_action = np.argmax(probs_pi)
            print(f'budget: {b}, optimal_value: {optimal_value}, optimal_action: {optimal_action}')
            min_value = min(min_value, optimal_value)
            max_value = max(max_value, optimal_value)
            avg_value += optimal_value
            if avg_pi is None:
                avg_pi = probs_pi
            else:
                avg_pi += probs_pi
        avg_pi /= 5
        probs_pi_dict[depth] = avg_pi/5
        avg_value /= 5

        mins[depth] = min_value
        maxs[depth] = max_value
        avgs[depth] = avg_value
    
    return mins,maxs,avgs

def plot_depth():
    cat_dynamics = catVsMonsters()
    mins_cat,maxs_cat, avgs_cat, iterations_cat = plot_graph_for_test_depth(cat_dynamics,min_visits=1)

    y_values =list(avgs_cat.values())
    x_values = [5,10,20,40,100,200]
    y_values = [x for x in y_values]
    plt.plot(x_values,y_values,label='avg', marker='o')
    plt.xlabel('depth')
    plt.ylabel('Average Return')
    plt.title('Grid World')
    min_values_grid = [x for x in list(mins_cat.values())]
    max_values_grid = [x for x in list(maxs_cat.values())]  
    plt.fill_between(x_values,min_values_grid,max_values_grid,alpha=0.5)
    plt.legend()
    plt.xscale('log')
    plt.grid(True)
    plt.show()

def get_policy(env,mcts,iterations=1000,depth=10):
    probs_pi_grid={}
    pi_grid={}
    for k in tqdm(range(25),desc='outer loop'):
        i = k // 5
        j = k % 5
        action_count = [0,0,0,0]
        for p in tqdm(range(10),desc='inner loop'):
            if (i,j) not in env.furniture and (i,j) != env.goal:
                env.current_state = (i,j)
                mcts.root = Node((i,j))
                probs_pi_grid[(i,j)],optimal_value = mcts.get_best_action(10000,min_visits=2000)
                optimal_action = np.argmax(probs_pi_grid[(i,j)])
                action_count[optimal_action] += 1
        if (i,j) not in env.furniture and (i,j) != env.goal:
            pi_grid[(i,j)] = action_count.index(max(action_count))
            print()
            print(f'state: {(i,j)}, action_count: {action_count}, pi[(i,j)]: {pi_grid[(i,j)]}')
            print(f'--------------------------------')
    
    return pi_grid

if __name__ == "__main__":
    plot_budget()
    plot_depth()

    env = catVsMonsters()
    mcts = MonteCarloSearchTree(env)
    pi_grid = get_policy(env,mcts,iterations=10000,depth=200)
    print_policy(pi_grid)

    
    



    
