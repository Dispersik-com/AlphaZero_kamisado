import unittest
from agent import Agent
from game_environment.kamisado_enviroment import KamisadoGame


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.game = KamisadoGame()
        self.agent = Agent(self.game)

    def test_find_best_move(self):
        self.game.make_move((7, 7), (3, 7))
        best_move = self.agent.find_best_move(self.game)
        print("best_move", best_move)

    # def test_load_models(self):
    #     policy_file = "research_results/net_train_result/vs UCB1-Tuned/policy_model.pth"
    #     value_file = "research_results/net_train_result/vs UCB1-Tuned/policy_model.pth"
    #     self.agent.load_models(policy_file, value_file)


if __name__ == '__main__':
    unittest.main()