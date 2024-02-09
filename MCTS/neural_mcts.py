from MCTS.mcts import MonteCarloTreeSearch


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
