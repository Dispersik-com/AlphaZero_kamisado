import unittest
from MCTS.neural_mcts import NeuralMonteCarloTreeSearch

from game_environment.kamisado_enviroment import KamisadoGame
from policy_value_networks import PolicyNet, ValueNet


class TestNeuralMonteCarloTreeSearch(unittest.TestCase):

    def setUp(self):
        self.game = KamisadoGame()
        self.policy_network = PolicyNet()
        self.value_network = ValueNet()

        self.mcts = NeuralMonteCarloTreeSearch(self.game, self.policy_network, self.value_network)

    def test_search(self):
        self.mcts.search(100)


if __name__ == "__main__":
    unittest.main()