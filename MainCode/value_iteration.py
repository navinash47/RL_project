import copy
import numpy as np
from grid_world import GridWorld
from catvsmonsters import catVsMonsters

class ValueIteration:
    def __init__(self, env):
        self.env = env
        self.V = {state: 0.0 for state in env.states}
        self.policy = {}
        self.gamma = 1
        self.delta = 0.001
    
    def calculate_value_iteration(self):
        """
        Calculate the value iteration for the given environment, 
        Input:
            env: The environment to calculate the value iteration for   
            gamma: The discount factor
            delta: The convergence threshold
        Output:
            V: The value function
            policy: The policy
        """
        i = 0
        while True:
            max_change = 0
            for state in self.env.states:
                # Skip terminal states
                if state == self.env.goal:
                    continue
                elif hasattr(self.env, 'furniture') and state in self.env.furniture:
                    # print(f'State: {state} is furniture')
                    continue
                elif hasattr(self.env, 'obstacles') and state in self.env.obstacles:
                    # print(f'State: {state} is obstacle')
                    continue
                
                v_old = self.V[state]
                v_new = float('-inf')
                
                # Find max Q-value over all actions
                for action in self.env.actions:
                    q_value = (self.env.p_intended * self.env.get_q_value_for_vi(state, action, "intended", self.gamma, self.V) +
                              self.env.p_right * self.env.get_q_value_for_vi(state, action, "right", self.gamma, self.V) +
                              self.env.p_left * self.env.get_q_value_for_vi(state, action, "left", self.gamma, self.V) +
                              self.env.p_stay * self.env.get_q_value_for_vi(state, action, "stay", self.gamma, self.V))
                    
                    if v_new < q_value:
                        v_new = q_value
                        self.policy[state] = action
                        
                self.V[state] = v_new
                max_change = max(max_change, abs(v_new - v_old))
                print(f'State: {state}, Iteration: {i}, Max Change: {max_change}')
                
            i += 1
            if max_change < self.delta:
                break
        
        return self.V, self.policy
    
    def get_q_values(self, state):
        """
        Get the q values for the given state
        Input:
            state: The state to get the q values for
        Output:
            max_q_value: The maximum q value for the given state
            optimal_action: The optimal action for the given state
        """
        q_values_for_state = {}
        # print(self.env.__class__.__name__)
        for action in self.env.actions:
            # if self.env.__class__.__name__ == "GridWorld":
            q_values_for_state[action] = self.env.p_intended * self.env.get_q_value_for_vi(
                state,
                    action,
                    course="intended",
                    gamma=self.gamma,
                V=self.V
            )
            q_values_for_state[action] += self.env.p_right * self.env.get_q_value_for_vi(
                state,
                    action,
                    course="right",
                    gamma=self.gamma,
                    V=self.V
                )
            q_values_for_state[action] += self.env.p_left * self.env.get_q_value_for_vi(
                state,
                    action,
                    course="left",
                    gamma=self.gamma,
                    V=self.V
                )
            q_values_for_state[action] += self.env.p_stay * self.env.get_q_value_for_vi(
                state,
                action,
                course="stay",
                gamma=self.gamma,
                V=self.V
                )
            
            
            
        # Find the action with maximum value
        optimal_action = max(q_values_for_state, key=q_values_for_state.get)
        max_q_value = q_values_for_state[optimal_action]
        # print(f'State: {state}, Action: {optimal_action}, Value: {max_q_value}')
        return optimal_action, max_q_value

    def print_policy(self):
        """
        Print the policy for the given environment in table format
        """
        for n, state in enumerate(self.env.states):
            i=n//5
            j=n%5
            # state=(i,j)
            if state == self.env.goal:
                print(f"G", end=' ')
            elif hasattr(self.env, 'furniture') and state in self.env.furniture:
                print(f"F", end=' ')
            elif hasattr(self.env, 'obstacles') and state in self.env.obstacles:
                print(f"O", end=' ')
            elif self.policy[state] == 'AU':
                print(f"↑", end=' ')
            elif self.policy[state] == 'AD':
                print(f"↓", end=' ')
            elif self.policy[state] == 'AL':
                print(f"←", end=' ')
            elif self.policy[state] == 'AR':
                print(f"→", end=' ')
            else:
                print(f" ", end=' ')
            if j==4:
                print()
    
    def print_value_function(self):
        """
        Print the value function for the given environment in table format
        """
        for n, state in enumerate(self.env.states):
            i=n//5
            j=n%5
            # state=(i,j)
            print(f"{self.V[state]:.4f}", end=' ')
            if j==4:
                print()

    
        
if __name__ == "__main__":
    env = GridWorld()
    # env = catVsMonsters()
    value_iteration = ValueIteration(env)
    value_iteration.calculate_value_iteration()
    value_iteration.print_policy()
    value_iteration.print_value_function()