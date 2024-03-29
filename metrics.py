import matplotlib.pyplot as plt
import matplotlib
import torch

import config

matplotlib.use(config.plot_backend)


def plot_metrics(metrics, xlabel="", ylabel="", title="", legend=True, save=True):
    """
    Plot metrics.

    Args:
        metrics (dict): Dictionary containing metrics data.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title for the plot.
        legend (bool): Whether to show legend or not.
        save (bool): Save image.
    """
    plt.figure(figsize=(10, 5))
    for label, data in metrics.items():
        plt.plot(data, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if legend:
        plt.legend()
    plt.show()

    if save:
        plt.savefig(title)


# def calculate_average_reward(total_reward, rewards):
#     """
#     Calculate the average reward from a list of rewards.
#
#     Args:
#         total_reward:
#         rewards:
#
#     Returns:
#         float: Average reward.
#     """
#
#     average_reward = total_reward / rewards
#     return average_reward


def evaluate_value_accuracy(model_estimations, true_values):
    """
    Evaluate the accuracy of state value estimations by comparing them with true values.

    Args:
        model_estimations (list): List of state value estimations made by the model.
        true_values (list): List of true state values.

    Returns:
        float: Accuracy of state value estimations.
    """
    total_absolute_error = sum(abs(model_value - true_value) for model_value, true_value in zip(model_estimations, true_values))
    total_absolute_error /= len(model_estimations)
    accuracy = 1 - total_absolute_error
    return accuracy


def evaluate_move_quality(model_predictions, expert_moves):
    """
    Evaluate the quality of generated moves by comparing them with expert moves.

    Args:
        model_predictions (list): List of softmax probability distributions predicted by the model.
                                  Each distribution corresponds to a move.
        expert_moves (list): List of expert moves. Each move is represented as a tuple of coordinates.

    Returns:
        float: Move quality score.
    """
    total_correct_predictions = 0

    for model_distribution, expert_move in zip(model_predictions, expert_moves):
        # Convert softmax distribution to a move (coordinate)
        predicted_move = torch.argmax(model_distribution).item()
        predicted_coordinates = (predicted_move // 8, predicted_move % 8)

        # Check if the predicted move matches any of the expert moves
        if predicted_coordinates in expert_moves:
            total_correct_predictions += 1
        else:
            total_correct_predictions -= 1

    move_quality = total_correct_predictions / len(model_predictions)
    return move_quality