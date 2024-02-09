import unittest

from MCTS.mcts import MonteCarloTreeSearch
from kamisado_environment.kamisado_enviroment import KamisadoGame


class TestMonteCarloTreeSearch(unittest.TestCase):

    def setUp(self):
        self.game = KamisadoGame()
        self.mcts = MonteCarloTreeSearch(self.game)

    def test_search_integration(self):
        self.mcts = MonteCarloTreeSearch(self.game)
        self.mcts.search(num_simulations=100)
