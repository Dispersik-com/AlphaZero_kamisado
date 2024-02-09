import torch
import torch.nn as nn
import torch.optim as optim


class PolicyNet(nn.Module):
    """
    A neural network model for policy estimation in a game.

    Attributes:
        conv1: First convolutional layer.
        conv2: Second convolutional layer.
        fc1: First fully connected layer.
        fc2: Second fully connected layer.
    """
    def __init__(self):
        """
        Initializes the PolicyNet model.
        """
        super(PolicyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 64)

    def forward(self, x):
        """
        Forward pass of the PolicyNet model.

        Args:
            x: Input tensor.

        Returns:
            policy: Output tensor representing the policy.
        """
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        policy = self.fc2(x)  # without applying softmax
        return policy

    def softmax_by_legal_moves(self, outputs, mask):
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



# policy_criterion = nn.CrossEntropyLoss()
# value_criterion = nn.MSELoss()
# policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
# value_optimizer = optim.Adam(value_net.parameters(), lr=0.001)
