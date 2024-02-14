import random
from MCTS.node import MCTSNode
from MCTS.tree_serializer import TreeSerializationMixin


class MonteCarloTreeSearch(TreeSerializationMixin):

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
            :param num_simulations:
            :param random_first_piece:
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
            node: The initial state of the game.

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

    @staticmethod
    def get_best_action(root):
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