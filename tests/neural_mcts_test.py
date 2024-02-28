import unittest
from MCTS.neural_mcts import NeuralMonteCarloTreeSearch

from game_environment.kamisado_enviroment import KamisadoGame
from policy_value_networks import PolicyNet, ValueNet


class TestNeuralMonteCarloTreeSearch(unittest.TestCase):

    def setUp(self):
        self.game = KamisadoGame()
        self.policy_network = PolicyNet()
        self.value_network = ValueNet()
        self.mcts = NeuralMonteCarloTreeSearch(self.game, self.policy_network, self.value_network,
                                               update_form_buffer=False)

    def test_search(self):
        self.mcts.search(100)

    def test_search_with_update_from_buffer(self):
        self.mcts = NeuralMonteCarloTreeSearch(self.game, self.policy_network, self.value_network,
                                               update_form_buffer=True, buffer_size=100)
        self.mcts.search(1000)


if __name__ == "__main__":
    unittest.main()