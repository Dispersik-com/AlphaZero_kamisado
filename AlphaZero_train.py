from MCTS.neural_mcts import NeuralMonteCarloTreeSearch, MonteCarloTreeSearch
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


def create_agent(game, use_gpu=False,
                 policy_learning_rate=0.001,
                 value_learning_rate=0.001,
                 player="Black",
                 update_form_buffer=True,
                 batch_size=1000):

    policy_net = PolicyNet(learning_rate=policy_learning_rate)
    value_net = ValueNet(learning_rate=value_learning_rate)

    if use_gpu:
        policy_net.cuda()
        value_net.cuda()

    neural_mcts = NeuralMonteCarloTreeSearch(game=game,
                                             player=player,
                                             policy_network=policy_net,
                                             value_network=value_net,
                                             update_form_buffer=update_form_buffer,
                                             buffer_size=batch_size)

    return neural_mcts


def self_play_and_train(epochs=10_000, num_simulations=1_000):
    game = KamisadoGame()

    agent_player = create_agent(game=game,
                                policy_learning_rate=0.01,
                                value_learning_rate=0.001,
                                player="Black",
                                update_form_buffer=True)

    for epoch in range(epochs):
        agent_player.search(num_simulations=num_simulations)

        print(f"Epochs: {epoch};",
              f"Games passed: {epoch * num_simulations};",
              f"Win rate: {agent_player.get_win_rate():.2f}",
              "=" * 100, sep="\n")

    losses = agent_player.get_losses()
    plot_losses(losses)

    agent_player.policy_network.save_model("policy_network_model_1.pth")
    agent_player.value_network.save_model("value_network_model_1.pth")


if __name__ == "__main__":
    self_play_and_train(epochs=100, num_simulations=10_000)
