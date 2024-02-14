import unittest

from MCTS.mcts import MonteCarloTreeSearch
from kamisado_environment.kamisado_enviroment import KamisadoGame
from MCTS.node import MCTSNode


class TestMonteCarloTreeSearch(unittest.TestCase):

    def setUp(self):
        self.game = KamisadoGame()
        self.mcts = MonteCarloTreeSearch(self.game)

    def test_simulate(self):
        node = self.mcts.root
        last_node, reward = self.mcts.simulate(node)
        self.assertIsInstance(last_node, MCTSNode)
        self.assertIsNotNone(reward)

    def test_select(self):
        node = self.mcts.root
        child_node = self.mcts.select(node)
        self.assertIsNotNone(child_node)

    def test_search_integration(self):
        self.mcts = MonteCarloTreeSearch(self.game)
        self.mcts.search(num_simulations=1000)
