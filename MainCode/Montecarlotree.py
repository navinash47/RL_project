import time
from tqdm import tqdm
import numpy as np
class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

class MonteCarloSearchTree:
    def __init__(self, env):
        self.env = env
        self.root = Node((0,0))
        self.depth_limit = 200
        self.c = 2
        self.gamma = 0.9

    def _selection(self, node):
        while len(node.children) > 0:
            ucb_values = [self._ucb(child, node.visits) for child in node.children]
            node = node.children[np.argmax(ucb_values)]
        return node

    def _ucb(self, child, parent_visits):
        if child.visits == 0:
            return float('inf')
        exploitation = child.value / child.visits
        exploration = self.c * np.sqrt(np.log(parent_visits) / child.visits)
        return exploitation + exploration

    def _expansion(self, node):
        for action in self.env.actions:
            next_state = self.env.get_next_state(node.state, action)
            child = Node(next_state, parent=node)
            node.children.append(child)
        return node

    def _simulation(self, node):
        # Save the original state
        original_state = self.env.current_state
        
        # Set environment to node's state
        self.env.current_state = node.state
        current_state = node.state
        total_reward = 0
        depth = 0
        
        while current_state not in [self.env.goal] and depth < self.depth_limit:
            action = np.random.choice(self.env.actions)
            next_state, reward, done = self.env.step(action)
            total_reward += (self.gamma ** depth) * reward
            current_state = next_state
            depth += 1
            if done:
                break
        
        # Restore the original state
        self.env.current_state = original_state
        return total_reward

    def _backpropagation(self, node, reward):
        while node is not None:
            node.visits += 1
            # Update to use average values instead of cumulative
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            reward = reward * self.gamma
            node = node.parent

    def get_best_action(self, iterations,min_visits=1):
        iteration_count = 0
        visits_completed= False
        for _ in range(iterations):
            leaf = self._selection(self.root)
            if leaf.state == self.env.goal:
                self._backpropagation(leaf, 0)
                continue
            # if leaf.visits == 0:
            leaf = self._expansion(leaf)
            simulation_result = self._simulation(leaf)
            self._backpropagation(leaf, simulation_result)
        
        # Calculate values for each action
        values = []
        for action in self.env.actions:
            next_state = self.env.get_next_state(self.root.state, action)
            for child in self.root.children:
                
                if child.state == next_state:
                    values.append(child.value)
                    break
        return np.exp(values)/np.sum(np.exp(values)), max(values)
