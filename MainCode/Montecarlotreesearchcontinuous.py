import numpy as np
from tqdm import tqdm

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

class MonteCarloTreeSearchContinuous:
    def __init__(self, env):
        self.env = env
        self.root = Node(env.reset()[0])
        self.depth_limit = 10
        self.c = 1.41
        self.gamma = 0.99

    def _selection(self, node):
        while node.children:
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
        for action in range(self.env.action_space.n):
            state = self.env.step(action)[0]
            child = Node(state, parent=node)
            node.children.append(child)
        return np.random.choice(node.children)

    def _simulation(self, node):
        state = node.state
        total_reward = 0
        done = False
        for _ in range(self.depth_limit):
            if done:
                break
            action = self.env.action_space.sample()
            state, reward, done, _, _ = self.env.step(action)
            total_reward += reward
        return total_reward

    def _backpropagation(self, node, reward):
        while node:
            node.visits += 1
            node.value += reward
            reward *= self.gamma
            node = node.parent

    def get_best_action(self, iterations):
        for _ in range(iterations):
            leaf = self._selection(self.root)
            # get the state and find whether it is in the goal state
            
            child = self._expansion(leaf)
            reward = self._simulation(child)
            self._backpropagation(child, reward)

        best_child = max(self.root.children, key=lambda c: c.visits)
        return self.root.children.index(best_child)

