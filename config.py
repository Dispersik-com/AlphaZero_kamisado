import torch

plot_backend = "TkAgg"

# set cuda if is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Learning rate for policy update
policy_learning_rate = 0.001

# Learning rate for value update
value_learning_rate = 0.001

# Flag indicating whether batch learning is enabled
batch_learning = True

# Size of data batch for single update
batch_size = 64

# Number of training epochs
epochs = 100

# Number of simulations in Monte Carlo method
num_simulations = 1000

# Flag indicating whether validation is enabled
validate = True

# Max number of validations per epoch
num_validations = 100

# Flag save models
save_models = True

# Model`s name
policy_save_filename = "policy_model.pth"

value_save_filename = "value_model.pth"

""" Opponent settings """

# Strategy for the opponent
opponent_strategy = "Epx3"  # Available strategies: UCB1, UCB1-Tuned, Epx3, ThompsonSampling

