import random
from collections import deque
from tqdm.autonotebook import tqdm
from MCTS.neural_mcts import NeuralMonteCarloTreeSearch
from policy_value_networks import PolicyNet, ValueNet
from game_environment.kamisado_enviroment import KamisadoGame
import config
from metrics import *

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    torch.cuda.set_device(device)


def create_agent(game, player="White"):

    policy_net = PolicyNet(learning_rate=config.policy_learning_rate)
    value_net = ValueNet(learning_rate=config.value_learning_rate)

    neural_mcts = NeuralMonteCarloTreeSearch(game=game,
                                             player=player,
                                             policy_network=policy_net,
                                             value_network=value_net,
                                             update_form_buffer=config.batch_learning,
                                             batch_size=config.batch_size)

    return neural_mcts


def self_play_and_train(epochs, num_simulations, num_validations, validate=False):
    game = KamisadoGame()
    agent_player = create_agent(game=game)

    win_rate_list = []
    value_accuracy = []
    eval_reward = []

    for _ in tqdm(range(epochs)):
        # agent_player.reward_by_player *= -1.

        agent_player.value_network.train()
        agent_player.policy_network.train()

        agent_player.search(num_simulations=num_simulations)
        agent_player.update_network()

        """  add metrics  """

        win_rate_list.append(agent_player.get_win_rate())

        if validate:
            agent_player.value_network.eval()
            agent_player.policy_network.eval()

            model_estimations = []
            true_values = []
            for i in range(num_validations):

                root = agent_player.root
                queue = deque([root])

                while queue:
                    current_node = queue.popleft()

                    with torch.no_grad():
                        # predict value
                        input_data = agent_player.convert_node_to_input(current_node)

                        model_estimations.append(agent_player.value_network(input_data).item())
                        node_value = torch.tanh(torch.tensor(current_node.value))
                        true_values.append(torch.tanh(node_value).item())

                    queue.extend(random.choice([current_node.children]))

            value_accuracy.append(evaluate_value_accuracy(model_estimations, true_values))

        eval_reward.append(calculate_average_reward(agent_player.total_reward, agent_player.count_rewards))

        print('-'*50,
              f'win rate: {agent_player.get_win_rate()}',
              f'B =  {agent_player.win_rate["Black"]}',
              f'W = {agent_player.win_rate["White"]}',
              sep="\n")


    # Plot metrics

    # losses = agent_player.get_losses()
    # plot_metrics(losses, xlabel="Epochs", ylabel="losses", title="Losses")

    plot_metrics({"Value accuracy": value_accuracy},
                 xlabel="Epochs", ylabel="accuracy", title="Value accuracy")
    plot_metrics({"Win Rate": win_rate_list},
                 xlabel="Epochs", ylabel="Win Rate", title="Win Rate")
    plot_metrics({"Rewards": eval_reward},
                 xlabel="Epochs", ylabel="reward", title="Evaluation rewards")

    if config.save_models:
        agent_player.policy_network.save_model(config.policy_save_filename)
        agent_player.value_network.save_model(config.value_save_filename)


if __name__ == "__main__":
    self_play_and_train(epochs=config.epochs,
                        num_simulations=config.num_simulations,
                        num_validations=config.num_validations)

    if device == "cuda":
        torch.cuda.empty_cache()
