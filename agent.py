import config
from MCTS.neural_mcts import NeuralMonteCarloTreeSearch
from policy_value_networks import PolicyNet, ValueNet


def create_agent(game, player="White"):

    policy_net = PolicyNet(learning_rate=config.policy_learning_rate)
    value_net = ValueNet(learning_rate=config.value_learning_rate)

    agent = NeuralMonteCarloTreeSearch(game=game,
                                             player=player,
                                             policy_network=policy_net,
                                             value_network=value_net,
                                             update_form_buffer=config.batch_learning,
                                             batch_size=config.batch_size)

    return agent