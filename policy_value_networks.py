import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import config
from abc import ABC, abstractmethod


class Network(ABC, nn.Module):

    def __init__(self, num_outputs=1, learning_rate=0.001):
        """
        Initializes the ValueNetLSTM model.
        """
        super(Network, self).__init__()
        self.rnn = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_outputs)
        self.learning_rate = learning_rate

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    @abstractmethod
    def forward(self, x):
        pass

    def create_mask(self, valid_actions):
        """
            Create a mask to restrict probabilities to legal actions.

            Args:
                valid_actions: List of valid action indices.

            Returns:
                mask: Tensor mask indicating valid actions.
            """
        indices_valid_actions = [i for i, x in enumerate(self.action_labels) if x in valid_actions]
        mask = torch.zeros(len(self.action_labels), device=config.device)
        mask[indices_valid_actions] = 1
        return mask

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
        model.load_state_dict(torch.load(file_path, map_location=torch.device(config.device)))
        model.eval()  # Set the model to evaluation mode
        return model


class PolicyNet(Network):

    def __init__(self, learning_rate=0.001):
        super().__init__(num_outputs=64, learning_rate=learning_rate)

        # set labels for output
        size = 8
        self.action_labels = [(x // size, x % size) for x in range(size * size)]

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x: Input tensor.

        Returns:
            out: Output tensor.
        """
        # LSTM layer
        x, _ = self.rnn(x)

        # Last time step output of LSTM
        x = x[:, -1]

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = x.view(-1, 256)
        policy = F.softmax(self.fc2(x), dim=1)
        return policy

    def batch_update(self, outputs, targets, reward: float):
        """
            Update the parameters of the policy network using an optimizer.

            Args:
                reward: List of rewards for moves.
                outputs: List of outputs of the policy network.
                targets: List of target labels for the outputs.

            Returns:
                losses
            """

        # Convert lists to tensors
        outputs_tensor = torch.stack(tuple(outputs), dim=0).to(config.device)
        rewards_tensor = torch.tensor(reward, requires_grad=True, device=config.device)

        # Create tensor for target labels
        target_indices = [torch.tensor(self.action_labels.index(target), dtype=torch.long) for target in targets]
        target_tensor = torch.tensor(target_indices, dtype=torch.long, device=config.device)
        one_hot_tensors = F.one_hot(target_tensor, num_classes=len(self.action_labels)).to(config.device)

        # Compute loss
        total_loss = torch.mean(F.cross_entropy(outputs_tensor.squeeze(1), one_hot_tensors.float()) * rewards_tensor)

        # Perform backward pass and optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss


class ValueNet(Network):

    def __init__(self, learning_rate=0.001):
        super().__init__(num_outputs=1, learning_rate=learning_rate)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x: Input tensor.

        Returns:
            out: Output tensor.
        """
        # LSTM layer
        x, _ = self.rnn(x)

        # Last time step output of LSTM
        x = x[:, -1]

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = x.view(-1, 256)
        value = F.sigmoid(self.fc2(x))
        return value

    def batch_update(self, outputs, rewards):
        """
        Batch update the parameters of the value network using an optimizer.

        Args:
            outputs: List of outputs of the value network.
            rewards: List of rewards for moves.

        Returns:
            loss
        """

        # Convert lists to tensors
        outputs_tensor = torch.stack(outputs).squeeze(1).to(config.device)
        rewards_tensor = torch.tensor(rewards, requires_grad=True).view(-1, 1).to(config.device)

        # Compute loss
        total_loss = F.mse_loss(outputs_tensor, rewards_tensor)

        # Perform backward pass and optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss
