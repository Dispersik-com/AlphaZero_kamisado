from MCTS.neural_mcts import NeuralMonteCarloTreeSearch, MonteCarloTreeSearch
from policy_value_networks import PolicyNet, ValueNet
from game_environment.kamisado_enviroment import KamisadoGame
import matplotlib.pyplot as plt
import matplotlib
import config
matplotlib.use('TkAgg')


def plot_losses(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses["policy_losses"], label="Policy Losses")
    plt.plot(losses["value_losses"], label="Value Losses")
    plt.plot()
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()


def create_agent(game, player="Black"):

    policy_net = PolicyNet(learning_rate=config.policy_learning_rate)
    value_net = ValueNet(learning_rate=config.value_learning_rate)

    if config.use_gpu:
        policy_net.cuda()
        value_net.cuda()

    neural_mcts = NeuralMonteCarloTreeSearch(game=game,
                                             player=player,
                                             policy_network=policy_net,
                                             value_network=value_net,
                                             update_form_buffer=config.batch_learning,
                                             buffer_size=config.batch_size)

    return neural_mcts


def self_play_and_train(epochs, num_simulations):
    game = KamisadoGame()
    agent_player = create_agent(game=game)

    for epoch in range(epochs):
        agent_player.search(num_simulations=num_simulations)

        print(f"Epochs: {epoch};",
              f"Games passed: {epoch * num_simulations};",
              f"Win rate: {agent_player.get_win_rate():.2f}",
              "=" * 100, sep="\n")

    losses = agent_player.get_losses()
    plot_losses(losses)

    if config.save_models:
        agent_player.policy_network.save_model(config.policy_save_filename)
        agent_player.value_network.save_model(config.value_save_filename)


if __name__ == "__main__":
    self_play_and_train(epochs=config.epochs,
                        num_simulations=config.num_simulations)
