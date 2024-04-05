import torch
from tqdm.autonotebook import tqdm
from MCTS.mcts import MonteCarloTreeSearch
from game_environment.kamisado_enviroment import KamisadoGame
from metrics import evaluate_value_accuracy, evaluate_move_quality, plot_metrics
from opponent import MCTSOpponent
from agent import Agent
import config


def play_and_train(epochs, num_simulations, num_validations, validate=True):

    # create game
    game = KamisadoGame()

    # create agent
    agent_player = Agent(game=game, player="White")

    # create opponent for agent
    mcts = MonteCarloTreeSearch(game=game, player="Black",
                                strategy=config.opponent_strategy)

    mcts_player = MCTSOpponent(game=game, agent_player=mcts)

    agent_player.set_opponent(mcts_player)

    # add metrics
    win_rate_list = []
    value_accuracy = []
    policy_accuracy = []
    eval_reward = []

    # train cycle
    for _ in tqdm(range(epochs)):

        agent_player.value_network.train()
        agent_player.policy_network.train()

        agent_player.search(num_simulations=num_simulations)
        agent_player.update_network()

        if validate:
            validation_data = agent_player.validate_searching(num_validations=num_validations)

            value_accuracy.append(evaluate_value_accuracy(validation_data["value_estimations"],
                                                          validation_data["true_values"]))

            policy_accuracy.append(evaluate_move_quality(validation_data["policy_estimations"],
                                                         validation_data["expert_moves"]))

        win_rate_list.append(agent_player.get_win_rate())

        eval_reward.append(agent_player.total_reward)

    # Plot metrics

    # losses = agent_player.get_losses()
    # plot_metrics(losses, xlabel="Epochs", ylabel="losses", title="Losses")

    plot_metrics({"Win Rate, %": win_rate_list},
                 xlabel="Epochs", ylabel="win rate, %", title="Win Rate")
    plot_metrics({"Rewards": eval_reward},
                 xlabel="Epochs", ylabel="rewards", title="Evaluation rewards")

    plot_metrics({"Value accuracy": value_accuracy},
                 xlabel="Epochs", ylabel="accuracy", title="Evaluation value accuracy")
    # plot_metrics({"Move quality": policy_accuracy},
    #              xlabel="Epochs", ylabel="accuracy", title="Evaluation move quality")

    if config.save_models:
        agent_player.policy_network.save_model(config.policy_save_filename)
        agent_player.value_network.save_model(config.value_save_filename)


if __name__ == "__main__":
    play_and_train(epochs=config.epochs,
                   num_simulations=config.num_simulations,
                   num_validations=config.num_validations,
                   validate=config.validate)

    if config.device == "cuda":
        torch.cuda.empty_cache()
