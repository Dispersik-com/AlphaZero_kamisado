# Learning rate for policy update
policy_learning_rate = 0.001

# Learning rate for value update
value_learning_rate = 0.001

# Flag indicating whether batch learning is enabled
batch_learning = False

# Size of data batch for single update
batch_size = 32

# Flag indicating whether GPU is utilized for computations
use_gpu = False

# Number of training epochs
epochs = 1000

# Number of simulations in Monte Carlo method
num_simulations = 10_000

# Flag save models
save_models = False

# Model`s name
policy_save_filename = "policy_model.pth"

value_save_filename = "value_model.pth"
