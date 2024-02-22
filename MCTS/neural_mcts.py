import copy

import torch

from MCTS.mcts import MonteCarloTreeSearch


class NeuralMonteCarloTreeSearch(MonteCarloTreeSearch):
    def __init__(self, game, policy_network, value_network, player="Black", exploration_weight=1.0):
        super().__init__(game, player=player,
                         exploration_weight=exploration_weight)
        # add policy and value networks
        self.policy_network = policy_network
        self.value_network = value_network

        self.current_reward = 0
        self.losses = {"policy_losses": [], "value_losses": []}

    def simulate(self, node):
        """
        Simulate a game starting from the given state.

        Args:
            node: The initial node of the game.

        Returns:
            The reward of the simulation.
        """
        current_node = node

        while not current_node.is_terminal():
            legal_actions = current_node.get_legal_actions()

            if not legal_actions:
                if not current_node.state.pass_move():
                    break
                continue

            # evaluate move probabilities using the policy_network
            # Prepare available moves in a suitable format for the neural network
            input_data = self.convert_node_to_input(current_node)

            # get move probabilities from the policy_network
            action_probs = self.policy_network.forward(input_data)

            # # apply mask by legal moves
            mask = self.policy_network.create_mask(legal_actions)
            action_probs_by_legal_moves = action_probs * mask.float()

            # choose a move according to the probabilities
            action_index = torch.argmax(action_probs_by_legal_moves).item()
            selected_action = self.policy_network.action_labels[action_index]

            current_node = current_node.expand(action=selected_action)

        # Evaluate the value of the final state using the value_network
        input_data = self.convert_node_to_input(current_node)
        value = self.value_network.forward(input_data)

        # format reward by player
        if current_node.is_winner("Black"):
            self.current_reward = 1.0 * self.reward_by_player
            print("Black")
        elif current_node.is_winner("White"):
            self.current_reward = -1.0 * self.reward_by_player
            print("White")

        # Return the state value as a reward
        return current_node, value

    def backpropagation(self, node, reward):
        """
        Update the node's visit count and value based on the result of a simulation.

        Args:
            node:
            reward: The reward of the simulation.
        """
        node.visits += 1
        node.value += reward

        input_data = self.convert_node_to_input(node)

        # Update the value network
        loss = self.value_network.update(reward, self.current_reward)
        self.losses["policy_losses"].append(loss.item())

        # Update the policy network
        loss = self.policy_network.update(input_data, node.action, reward)
        self.losses["value_losses"].append(loss.item())

        if node.parent and node.parent.action is not None:
            self.backpropagation(node.parent, reward)

    def get_losses(self, reset_losses=False):
        losses = self.losses
        if reset_losses:
            self.losses = {"policy_losses": [], "value_losses": []}
        return losses

    @staticmethod
    def convert_node_to_input(node):
        """
        Convert the node to a format suitable for input to the neural network.

        Args:
            node: The MCTS node.

        Returns:
            Input data for the neural network.
        """
        state = node.state.game_board.tolist()

        piece_to_idx = {'B-O': 0, 'B-A': 1, 'B-V': 2, 'B-P': 3,
                        'B-Y': 4, 'B-R': 5, 'B-G': 6, 'B-B': 7,
                        'W-B': 8, 'W-G': 9, 'W-R': 10, 'W-Y': 11,
                        'W-P': 12, 'W-V': 13, 'W-A': 14, 'W-O': 15}

        state_copy = copy.deepcopy(state)

        for i, row in enumerate(state_copy):
            for j, cell in enumerate(row):
                if str(cell) in piece_to_idx.keys():
                    state_copy[i][j] = piece_to_idx[str(cell)]

        # create tensor
        tensor = torch.tensor(state_copy, dtype=torch.float32).view(1, 64).clone().detach()

        return tensor
