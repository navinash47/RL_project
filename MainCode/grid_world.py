import numpy as np

class GridWorld:
    def __init__(self):
        # Grid dimensions
        self.rows = 5
        self.cols = 5
        self.states = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        # Define states
        self.obstacles = [(2, 2), (3, 2)]  # Obstacle states
        self.water = (4, 2)                # Water state
        self.goal = (4, 4)                 # Goal state
        
        # Define actions
        self.actions = ['AU', 'AD', 'AL', 'AR']  # Up, Down, Left, Right
        
        # Action probabilities
        self.p_intended = 0.8    # Probability of moving in intended direction
        self.p_right = 0.05
        self.p_left = 0.05
        self.p_stay = 0.10      # Probability of staying in place
        
        # Current state
        self.current_state = (0, 0)  # Start at State 1
        
    def is_valid_state(self, state):
        """Check if state is valid (within bounds and not obstacle)"""
        r, c = state
        return (0 <= r < self.rows and 
                0 <= c < self.cols and 
                state not in self.obstacles)
    
    def get_next_state(self, state, action):
        """Get next state based on current state and action"""
        r, c = state
        
        # Define movement directions (up, down, left, right)
        movements = {
            'AU': (-1, 0),
            'AD': (1, 0),
            'AL': (0, -1),
            'AR': (0, 1)
        }
        
        # Get intended movement
        dr, dc = movements[action]
        new_r, new_c = r + dr, c + dc
        
        # Check if move is valid
        if self.is_valid_state((new_r, new_c)):
            return (new_r, new_c)
        return state  # Stay in current state if invalid move
    
    def step(self, action):
        """Take a step in the environment"""
        if action not in self.actions:
            raise ValueError("Invalid action")
            
        # Determine actual movement based on probabilities
        p = np.random.random()
        
        if p < self.p_stay:  # 10% chance to stay
            actual_action = None
        elif p < self.p_stay + self.p_intended:  # 80% chance for intended action
            actual_action = action
        elif p < self.p_stay + self.p_intended + self.p_right:
            if action == 'AU':
                actual_action = 'AR'
            elif action == 'AD':
                actual_action = 'AL'
            else:
                actual_action = 'AU'
        else:
            if action == 'AU':
                actual_action = 'AL'
            elif action == 'AD':
                actual_action = 'AR'
            else:
                actual_action = 'AD'
        
        # Update state
        if actual_action:
            self.current_state = self.get_next_state(self.current_state, actual_action)
        
        # Calculate reward
        if self.current_state == self.goal:
            reward = 10
        elif self.current_state == self.water:
            reward = -10
        else:
            reward = 0
        
        # Check if episode is done
        done = self.current_state == self.goal
        
        return self.current_state, reward, done
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_state = (0, 0)
        return self.current_state

    def get_q_value_for_vi(self, state, action, course, gamma, V):
        """Get the q value for the given state and action"""
        if course == "intended":
            next_state, reward, _ = self.step(action)
            return reward + gamma * V[next_state]
        elif course == "right":
            next_state, reward, _ = self.step(self.right_of(action))
            return reward + gamma * V[next_state]
        elif course == "left":
            next_state, reward, _ = self.step(self.left_of(action))
            return reward + gamma * V[next_state]
        elif course == "stay":
            # Calculate reward for staying in the same state
            if state == self.goal:
                reward = 10
            elif state == self.water:
                reward = -10
            else:
                reward = 0
            return reward + gamma * V[state]
    def right_of(self,action):
        """Get the right of the given state"""
        if action == 'AU':
            return 'AR'
        elif action == 'AD':
            return 'AL'
        elif action == 'AL':
            return 'AU'
        elif action == 'AR':
            return 'AD'
    def left_of(self,action):
        """Get the left of the given state"""
        if action == 'AU':
            return 'AL'
        elif action == 'AD':
            return 'AR'
        elif action == 'AL':
            return 'AD'
        elif action == 'AR':
            return 'AU'



def get_policy(env,mcts,iterations=10000,depth=200):

    from tqdm import tqdm
    env_grid = GridWorld()
    mcts_grid = MonteCarloSearchTree(env_grid)
    probs_pi_grid={}
    pi_grid={}
    for k in tqdm(range(25),desc='outer loop'):
        i = k // 5
        j = k % 5
        action_count = [0,0,0,0]
        for p in tqdm(range(10),desc='inner loop'):
            if (i,j) not in env.furniture and (i,j) != env.goal:
                env_grid.current_state = (i,j)
            mcts_grid.root = Node((i,j))
            probs_pi_grid[(i,j)],optimal_value = mcts_grid.get_best_action(10000,min_visits=2000)
            optimal_action = np.argmax(probs_pi_grid[(i,j)])
            action_count[optimal_action] += 1
        if (i,j) not in env_grid.obstacles and (i,j) != env_grid.goal:
            pi_grid[(i,j)] = action_count.index(max(action_count))
            print()
            print(f'state: {(i,j)}, action_count: {action_count}, pi[(i,j)]: {pi_grid[(i,j)]}')
            print(f'--------------------------------')
    return pi_grid

def print_policy_grid(policy):
    for i in range(5):
        for j in range(5):
            state = (i, j)
            if state == env_grid.goal:
                print("G", end=" ")
                continue
            elif state in env_grid.obstacles:
                print("O", end=" ")
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
    grid_dynamics = GridWorld()
    mins_grid,maxs_grid, avgs_grid, iterations_grid = plot_graph_for_budget(grid_dynamics,min_visits=1)

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
    grid_dynamics = GridWorld()
    mins_grid,maxs_grid, avgs_grid = plot_graph_for_test_depth(grid_dynamics,min_visits=1)

    y_values =list(avgs_grid.values())
    x_values = [5,10,20,40,100,200]
    y_values = [x for x in y_values]
    plt.plot(x_values,y_values,label='avg', marker='o')
    plt.xlabel('depth')
    plt.ylabel('Average Return')
    plt.title('Grid World')
    min_values_grid = [x for x in list(mins_grid.values())]
    max_values_grid = [x for x in list(maxs_grid.values())]  
    plt.fill_between(x_values,min_values_grid,max_values_grid,alpha=0.5)
    plt.legend()
    plt.xscale('log')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_budget()
    plot_depth()
    
    env = GridWorld()
    mcts = MonteCarloSearchTree(env)
    pi_grid = get_policy(env,mcts,iterations=10000,depth=200)
    print_policy_grid(pi_grid)
    


