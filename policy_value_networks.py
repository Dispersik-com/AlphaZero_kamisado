import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class PolicyNet(nn.Module):
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

    def forward(self, x, legal_moves=None):
        """
        Forward pass of the PolicyNet model.

        Args:
            x: Input tensor.
            legal_moves: A list of legal moves. If None, all moves are considered legal.

        Returns:
            policy: Output tensor representing the policy.
        """
        if legal_moves is None:
            legal_moves = self.action_labels

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        policy = self.fc2(x)  # without applying softmax

        # apply mask and softmax by legal moves
        mask = self.create_mask(legal_moves)
        policy_by_legal_moves = self.softmax_by_legal_moves(policy, mask)
        return policy_by_legal_moves

    def update(self, output, target, reward: int):
        """
        Update the parameters of the policy network using an optimizer.

        Args:
            reward: reward of moves
            output: The output of the policy network.
            target: The target labels for the output.
            learning_rate: The learning rate for the optimizer.

        Returns:
            None
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        loss = F.cross_entropy(output, target) * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


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

    @staticmethod
    def softmax_by_legal_moves(outputs, mask):
        """
        Applies softmax function to the outputs considering only legal moves.

        Args:
            outputs: Output tensor from the network.
            mask: Mask indicating legal moves.

        Returns:
            masked_outputs: Output tensor with softmax applied only to legal moves.
        """
        # Apply the mask to the network outputs
        masked_outputs = outputs * mask.float()
        # Apply softmax only to legal moves
        masked_outputs = torch.softmax(masked_outputs, dim=0)
        return masked_outputs


class ValueNet(nn.Module):
    """
    A neural network model for estimating the value of a game state.

    Attributes:
        conv1: First convolutional layer.
        conv2: Second convolutional layer.
        fc1: First fully connected layer.
        fc2: Second fully connected layer.
    """
    def __init__(self):
        """
        Initializes the ValueNet model.
        """
        super(ValueNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)    # output for state value

    def forward(self, x):
        """
        Forward pass of the ValueNet model.

        Args:
            x: Input tensor.

        Returns:
            value: Output tensor representing the estimated value of the state.
        """
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        value = torch.tanh(self.fc2(x))  # output for state value, in range [-1, 1]
        return value

    def update(self, output, reward: int, learning_rate=0.001):
        """
        Update the parameters of the value network using an optimizer.

        Args:
            reward: reward of moves
            output: The output of the value network.
            learning_rate: The learning rate for the optimizer.

        Returns:
            None
        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss = F.mse_loss(output, reward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
