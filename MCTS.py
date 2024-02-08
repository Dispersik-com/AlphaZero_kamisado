import math
import random
from collections import deque

import numpy as np
import json


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = self.get_legal_actions()

    def get_legal_actions(self):
        """
        Get the legal actions available in the current state.

        Returns:
            A list of tuples representing legal actions.
        """
        return self.state.get_legal_moves()

    def is_fully_expanded(self):
        """
        Check if all possible actions have corresponding child nodes.

        Returns:
            True if fully expanded, False otherwise.
        """
        return not bool(self.untried_actions)

    def select_child(self, exploration_weight=1.0, strategy="UCB1"):
        """
        Select a child node based on a selection strategy (e.g., UCB1).

        Args:
            exploration_weight: The weight for exploration in the selection strategy.

        Returns:
            The selected child node.
        """

        if not self.children:
            return self

        if strategy == "UCB1":
            selection_function = lambda child: child.value / child.visits + exploration_weight * math.sqrt(math.log(self.visits) / child.visits)

        elif strategy == "UCB1-Tuned":
            selection_function = lambda child: child.value / child.visits + exploration_weight * math.sqrt(math.log(self.parent.visits) / (2 * child.visits))
        
        # TODO: To be implemented are other strategies: Exp3 and Thompson Sampling.

        else:
            raise ValueError("Invalid strategy specified")

        selected_child = max(self.children, key=selection_function)

        return selected_child
    
    def expand(self, action=None):
        """
        Expand the node by adding a child with an untried action.

        Returns:
            The created child node.
        """
        if action is None:
            # If no specific action is provided for expanding the node:

            # Check if the node is fully expanded (all possible actions are available)
            if self.is_fully_expanded():
                # selected_child = random.choice(self.children)
                selected_child = self.select_child()
                return selected_child

            # Get a list of legal actions that can be applied to the current node
            legal_moves_by_piece = self.get_legal_actions()

            # Choose one of the untried actions that has not been used in this node yet
            action = list(set(self.untried_actions) & set(legal_moves_by_piece)).pop()
            # Remove the chosen action from the list of untried actions
            self.untried_actions.pop(self.untried_actions.index(action))

        # Perform the action in a copy of the current state to create a new state
        new_state = self.state.copy_and_apply_action(action)
        child_node = MCTSNode(new_state, parent=self, action=action)

        # Check for duplicates among the child nodes by states
        new_state = child_node.state
        if any(hash(child.state) == hash(new_state) for child in self.children):
            return child_node

        self.children.append(child_node)
        return child_node

    def backpropagation(self, reward):
        """
        Update the node's visit count and value based on the result of a simulation.

        Args:
            reward: The reward of the simulation.
        """
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagation(reward)

    def is_terminal(self):
        """
        Check if the current state is a terminal state.

        Returns:
            True if terminal, False otherwise.
        """
        self.state.check_winner()
        has_action = bool(len(self.get_legal_actions()) != 0)

        return self.state.winner is not None and has_action

    def is_winner(self, player):
        """
        Check if the current state is a winning state for the specified player.

        Args:
            player: The player for whom to check the winning state.

        Returns:
            True if winning, False otherwise.
        """
        self.state.check_winner()

        return self.state.winner == player


class MonteCarloTreeSearch:
    def __init__(self, game, exploration_weight=1.0):
        """
        Initialize the Monte Carlo Tree Search.

        Args:
            game: The game environment.
            exploration_weight: The weight for exploration in the selection strategy.
        """
        self.game = game
        self.exploration_weight = exploration_weight
        self.root = MCTSNode(self.game)  # root of MCTS

    def search(self, num_simulations, random_first_piece=True):
        """
        Perform Monte Carlo Tree Search.

        Args:
            num_simulations: The number of simulations to run.

        Returns:
            The selected action.
        """

        for i in range(num_simulations):

            if random_first_piece:
                self.root.state.set_first_piece()
            else:
                piece_color = self.game.color_dict[i]
                self.root.state.set_first_piece(piece_color)

            # step 1: select
            selected_node = self.select(self.root)

            # step 2: expanding
            if not selected_node.is_terminal():
                expanded_node = selected_node.expand()

                # step 3: simulation
                simulation_result = self.simulate(expanded_node)
                last_node, reward = simulation_result

                # step 4: backpropagation
                last_node.backpropagation(reward)

        # select the best action based on root statistics
        best_action = self.get_best_action(self.root)
        return best_action

    def simulate(self, node):
        """
        Simulate a game starting from the given state.

        Args:
            state: The initial state of the game.

        Returns:
            The reward of the simulation.
        """
        current_node = node

        while not current_node.is_terminal():
            legal_actions = current_node.get_legal_actions()

            if not legal_actions:
                if not current_node.state.pass_move():
                    break
                continue

            action = random.choice(legal_actions)
            current_node = current_node.expand(action=action)

        if current_node.is_winner("Black"):
            print("Black player wins")
            return current_node, 1.0  # Rewards if the Black win
        elif current_node.is_winner("White"):
            print("White player wins")
            return current_node, -1.0  # Rewards if the White win

    def select(self, node):
        """
        Select a child node based on the selection strategy (e.g., UCB1).

        Args:
            node: The current node to start the selection.

        Returns:
            The selected child node.
        """
        return node.select_child(self.exploration_weight)

    def get_best_action(self, root):
        """
        Select the best action based on root statistics.

        Args:
            root: The root node of the MCTS tree.

        Returns:
            The best action.
        """
        if not root.children:
            return None

        best_child = max(root.children, key=lambda child: child.value)

        return best_child.action

    def to_dict(self):
        """
        Convert the MCTS tree rooted at the given node to a dictionary using BFS.

        Args:
            node: The root node of the tree.

        Returns:
            A dictionary representation of the MCTS tree.
        """
        if self.root.children is None:
            return None

        tree_data = {}
        queue = deque(self.root.children)

        while queue:
            current_node = queue.popleft()
            children_data = [child for child in current_node.children]
            tree_data[str(hash(current_node.state))] = {
                'state': str(current_node.state.game_board.tolist()),
                'parent': str(hash(current_node.parent.state)),
                'action': current_node.action,
                'children': list(map(lambda x: str(hash(x.state)), children_data)),
                'visits': current_node.visits,
                'value': current_node.value,
                # 'untried_actions': current_node.untried_actions
            }

            queue.extend(current_node.children)

        return tree_data

    def save_tree(self, filename, indent=3):
        """
        Save the entire MCTS tree to a file using BFS-based serialization.

        Args:
            filename: The name of the file to save the tree.
        """
        tree_data = self.to_dict()

        # tree_data = convert_keys(tree_data)
        with open(filename, 'w') as file:
            json.dump(tree_data, file, indent=indent)

    def load_tree(self, filename):
        """
        Load the MCTS tree from a file.

        Args:
            filename: The name of the file containing the tree.

        Returns:
            The dictionary representation of the loaded MCTS tree.
        """
        with open(filename, 'r') as file:
            tree_data = json.load(file)
        return tree_data


class NeuralMonteCarloTreeSearch(MonteCarloTreeSearch):
    def __init__(self, game, policy_network, value_network, exploration_weight=1.0):
        super().__init__(game, exploration_weight)
        # add policy and value networks
        self.policy_network = policy_network
        self.value_network = value_network

    def simulate(self, node):
        """
        Simulate a game starting from the given state.

        Args:
            node: The initial node of the game.

        Returns:
            The reward of the simulation.
        """
        pass

    def convert_node_to_input(self, node):
        """
        Convert the node to a format suitable for input to the neural network.

        Args:
            node: The MCTS node.

        Returns:
            Input data for the neural network.
        """
        pass