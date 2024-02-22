import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class SaveLoadInterface:

    def save_model(self, file_path):
        """
        Saves the model parameters to a file.

        Args:
            file_path (str): The path to the file where the model parameters will be saved.
        """
        torch.save(self.state_dict(), file_path)

    @classmethod
    def load_model(cls, file_path):
        """
        Creates a new instance of the model and loads parameters from a file.

        Args:
            file_path (str): The path to the file from which the model parameters will be loaded.

        Returns:
            PolicyNet: The loaded model.
        """
        model = cls()
        model.load_state_dict(torch.load(file_path))
        model.eval()  # Set the model to evaluation mode
        return model


class PolicyNet(nn.Module, SaveLoadInterface):
    """
    A neural network model for policy estimation in a game.

    Attributes:
        conv1: First convolutional layer.
        conv2: Second convolutional layer.
        fc1: First fully connected layer.
        fc2: Second fully connected layer.
    """
    def __init__(self, learning_rate=0.001):
        """
        Initializes the PolicyNet model.
        """
        super(PolicyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 64)
        self.learning_rate = learning_rate

        # set labels for output
        size = 8
        self.action_labels = [(x // size, x % size) for x in range(size*size)]

    def forward(self, x):
        """
        Forward pass of the PolicyNet model.

        Args:
            x: Input tensor.

        Returns:
            policy: Output tensor representing the policy.
        """

        x = x.view(1, 1, 8, 8)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        policy = torch.softmax(self.fc2(x), dim=0)

        return policy

    def update(self, output, target, reward: float):
        """
        Update the parameters of the policy network using an optimizer.

        Args:
            reward: reward of moves
            output: The output of the policy network.
            target: The target labels for the output.

        Returns:
            None
        """
        one_hot_target = self.get_one_hot_target(target).clone().detach().requires_grad_(True)
        reward = torch.tensor([reward]).view(1, 1)

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        loss = F.cross_entropy(output, one_hot_target) * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def create_mask(self, valid_actions):
        """
            Create a mask to restrict probabilities to legal actions.

            Args:
                valid_actions: List of valid action indices.

            Returns:
                mask: Tensor mask indicating valid actions.
            """
        indices_valid_actions = [i for i, x in enumerate(self.action_labels) if x in valid_actions]
        mask = torch.zeros(len(self.action_labels))
        mask[indices_valid_actions] = 1
        return mask

    def get_one_hot_target(self, target: tuple):
        """
        Generates a one-hot representation for a given target action.

        Args:
            target (tuple): A tuple representing the coordinates of the target action.

        Returns:
            torch.Tensor: A one-dimensional tensor representing the one-hot encoding of the target action.
        """
        num_classes = len(self.action_labels)

        one_hot_label = torch.zeros(num_classes)
        action_index = self.action_labels.index(target)

        one_hot_label[action_index] = 1.

        tensor = one_hot_label.view(1, 64).clone().detach()

        return tensor


class ValueNet(nn.Module, SaveLoadInterface):
    """
    A neural network model for estimating the value of a game state.

    Attributes:
        conv1: First convolutional layer.
        conv2: Second convolutional layer.
        fc1: First fully connected layer.
        fc2: Second fully connected layer.
    """
    def __init__(self, learning_rate=0.001):
        """
        Initializes the ValueNet model.
        """
        super(ValueNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)    # output for state value

        self.learning_rate=learning_rate

    def forward(self, x):
        """
        Forward pass of the ValueNet model.

        Args:
            x: Input tensor.

        Returns:
            value: Output tensor representing the estimated value of the state.
        """
        x = x.view(1, 1, 8, 8)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        value = torch.tanh(self.fc2(x))  # output for state value, in range [-1, 1]
        return value

    def update(self, output, reward: float):
        """
        Update the parameters of the value network using an optimizer.

        Args:
            reward: reward of moves
            output: The output of the value network.

        Returns:
            None
        """
        reward_tensor = torch.tensor([reward], dtype=torch.float32).view(1, 1)
        reward_tensor = reward_tensor.clone().detach()
        output = output.clone().detach().requires_grad_(True)

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        loss = F.mse_loss(output, reward_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss
