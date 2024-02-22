from MCTS.neural_mcts import NeuralMonteCarloTreeSearch
from policy_value_networks import PolicyNet, ValueNet
from game_environment.kamisado_enviroment import KamisadoGame
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def plot_losses(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses["policy_losses"], label="Policy Losses")
    plt.plot(losses["value_losses"], label="Value Losses")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()


def create_agent(game, use_gpu=False):
    policy_net = PolicyNet()
    value_net = ValueNet()

    if use_gpu:
        policy_net.cuda()
        value_net.cuda()

    neural_mcts = NeuralMonteCarloTreeSearch(game=game,
                                             policy_network=policy_net,
                                             value_network=value_net,
                                             exploration_weight=1.0)

    return neural_mcts


def self_play_and_train(epochs=10_000, num_simulations=1_000):
    game = KamisadoGame()
    agent_player = create_agent(game=game)

    for epoch in range(epochs):
        agent_player.search(num_simulations=num_simulations)

        print(f"Epochs: {epoch};",
              "=" * 100, sep="\n")

    losses = agent_player.get_losses()
    plot_losses(losses)


if __name__ == "__main__":
    self_play_and_train(epochs=10, num_simulations=100)
