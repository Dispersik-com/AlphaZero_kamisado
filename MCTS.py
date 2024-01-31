import math
import random
import numpy as np
from kamisado_environment import KamisadoEnvironment

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


    def select_child(self, exploration_weight, strategy="UCB1"):
        """
        Select a child node based on a selection strategy (e.g., UCB1).

        Args:
            exploration_weight: The weight for exploration in the selection strategy.

        Returns:
            The selected child node.
        """

        if not self.children:
            return None

        if strategy == "UCB1":
            selection_function = lambda child: child.value / child.visits + exploration_weight * math.sqrt(math.log(self.visits) / child.visits)

        elif strategy == "UCB1-Tuned":
            selection_function = lambda child: child.value / child.visits + exploration_weight * math.sqrt(math.log(self.parent.visits) / (2 * child.visits))
        
        # TODO: To be implemented are other strategies: Exp3 and Thompson Sampling.

        else:
            raise ValueError("Invalid strategy specified")

        selected_child = max(self.children, key=selection_function)

        return selected_child
    
    def expand(self):
        """
        Expand the node by adding a child with an untried action.

        Returns:
            The created child node.
        """
        # Select an untried action
        action = self.untried_actions.pop()
        
        # Perform the action in a copy of the current state to create a new state
        new_state = self.state.copy_and_apply_action(action)
        child = MCTSNode(new_state, parent=self, action=action)
        self.children.append(child)
        return child
    
    def get_child_by_action(self, action):
        """
        Get the child node corresponding to the specified action.

        Args:
            action: The action for which to get the child node.

        Returns:
            The child node corresponding to the specified action, or None if not found.
        """
        for child in self.children:
            if child.action == action:
                return child

        return None

    def backpropagate(self, reward):
        """
        Update the node's visit count and value based on the result of a simulation.

        Args:
            reward: The reward of the simulation.
        """
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)

    def is_terminal(self):
        """
        Check if the current state is a terminal state.

        Returns:
            True if terminal, False otherwise.
        """
        winner = self.state.check_winner()

        return winner is not None
    
    
    def is_winner(self, player):
        """
        Check if the current state is a winning state for the specified player.

        Args:
            player: The player for whom to check the winning state.

        Returns:
            True if winning, False otherwise.
        """
        winner = self.state.check_winner()
        if winner:
            return winner == player
        else:
            return False


class MonteCarloTreeSearch:
    def __init__(self, game, policy_network, value_network, exploration_weight=1.0):
        """
        Initialize the Monte Carlo Tree Search.

        Args:
            game: The game environment.
            policy_network: The neural network for policy estimation.
            value_network: The neural network for value estimation.
            exploration_weight: The weight for exploration in the selection strategy.
        """
        self.game = game
        self.policy_network = policy_network
        self.value_network = value_network
        self.exploration_weight = exploration_weight
        self.root = None  # root of MCTS


    def search(self, num_simulations):
        """
        Perform Monte Carlo Tree Search.

        Args:
            num_simulations: The number of simulations to run.

        Returns:
            The selected action.
        """
        for _ in range(num_simulations):
            # step 1: select
            selected_node = self.select(self.root)

            # step 2: expanding
            if not selected_node.is_terminal():
                expanded_node = selected_node.expand()

                # step 3: simulation
                simulation_result = self.simulate(expanded_node)

                # step 4: backpropagate
                expanded_node.backpropagate(simulation_result)

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
          legal_actions = current_node.state.get_legal_actions()
          if not legal_actions:
                break

          action = random.choice(legal_actions)
          current_node = current_node.get_child_by_action(action)

        if current_node.is_winner("Black"):
            return 1.0  # Rewards if the Black win
        elif current_node.is_winner("White"):
            return -1.0  # Rewards if the White win
        else:
            return 0.0 # Rewards if the game ends in a draw or is still ongoing


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
