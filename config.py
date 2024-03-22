import torch

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
epochs = 10

# Number of simulations in Monte Carlo method
num_simulations = 100

# Flag indicating whether validation is enabled
validate = True

# Max number of validations per epoch
num_validations = 50

# Flag save models
save_models = False

# Model`s name
policy_save_filename = "policy_model.pth"

value_save_filename = "value_model.pth"

""" Opponent settings """

# Strategy for the opponent
opponent_strategy = "ThompsonSampling"  # Available strategies: UCB1, UCB1-Tuned, Epx3, ThompsonSampling

