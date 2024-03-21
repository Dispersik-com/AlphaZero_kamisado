import torch

from MCTS.mcts import MonteCarloTreeSearch


class NeuralMonteCarloTreeSearch(MonteCarloTreeSearch):
    def __init__(self, game, policy_network, value_network, player="Black", update_form_buffer=True, batch_size=32):
        super().__init__(game, player=player)
        # add policy and value networks
        self.policy_network = policy_network
        self.value_network = value_network

        self.current_reward = 0
        self.losses = {"policy_losses": [], "value_losses": []}
        self.update_form_buffer = update_form_buffer

        self.batch_size = batch_size
        self.buffer = {
                       "states_data": [],
                       "selected_action_data": [],
                       "node_value_data": []
                       }

        # metrics
        self.win_rate = {"Black": 0, "White": 0}
        self.total_reward = 0
        self.count_rewards = 0

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
            action_probs = self.policy_network(input_data)

            # apply mask by legal moves
            mask = self.policy_network.create_mask(legal_actions)
            action_probs_by_legal_moves = action_probs * mask.float()

            # choose a move according to the probabilities
            action_index = torch.argmax(action_probs_by_legal_moves).item()
            selected_action = self.policy_network.action_labels[action_index]

            # add data to buffer

            self.buffer["states_data"].append(input_data)
            self.buffer["selected_action_data"].append(selected_action)

            # get real value data and normalized
            normalized_node_value = torch.tanh(torch.tensor(current_node.value))
            self.buffer["node_value_data"].append(normalized_node_value)

            current_node = current_node.expand(action=selected_action)

        # format reward by player
        if current_node.is_winner("Black"):
            self.current_reward = 1.0 * self.reward_by_player
            self.win_rate["Black"] += 1
        elif current_node.is_winner("White"):
            self.current_reward = -1.0 * self.reward_by_player
            self.win_rate["White"] += 1

        self.total_reward += self.current_reward
        self.count_rewards += 1
        # Return the state value as a reward
        return current_node, self.current_reward

    def update_network(self):

        if self.update_form_buffer:
            shift = 0
            while True:
                states = self.buffer["states_data"][shift:self.batch_size + shift]
                selected_actions = self.buffer["selected_action_data"][shift:self.batch_size + shift]
                node_values = self.buffer["node_value_data"][shift:self.batch_size + shift]

                if len(states) == 0:
                    break

                values = []
                for s in states:
                    values.append(self.value_network(s))

                loss = self.value_network.batch_update(values, node_values)
                self.losses["value_losses"].append(loss.item())

                # Update the policy network
                loss = self.policy_network.batch_update(states, selected_actions, node_values)
                self.losses["policy_losses"].append(loss.item())

                shift += self.batch_size

        else:
            for i in range(len(self.buffer["states_data"])):

                state = self.buffer["states_data"][i]
                selected_action = self.buffer["selected_action_data"][i]
                node_value = self.buffer["node_value_data"][i]

                value = self.value_network(state)
                loss = self.value_network.update(value, node_value)
                self.losses["value_losses"].append(loss.item())

                # Update the policy network
                loss = self.policy_network.update(state, selected_action, node_value)
                self.losses["policy_losses"].append(loss.item())

        # clean buffer
        self.buffer = {"states_data": [], "selected_action_data": [], "node_value_data": []}

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

        piece_to_idx = {'B-O': 1, 'B-A': 2, 'B-V': 3, 'B-P': 4,
                        'B-Y': 5, 'B-R': 6, 'B-G': 7, 'B-B': 8,
                        'W-B': 9, 'W-G': 10, 'W-R': 11, 'W-Y': 12,
                        'W-P': 13, 'W-V': 14, 'W-A': 15, 'W-O': 16}

        for i, row in enumerate(state):
            for j, cell in enumerate(row):
                if str(cell) in piece_to_idx.keys():
                    state[i][j] = piece_to_idx[str(cell)]

        # create tensor
        tensor = torch.tensor(state, dtype=torch.float32, requires_grad=True).view(-1).detach()

        return tensor

    def get_win_rate(self):
        total_games = self.win_rate["Black"] + self.win_rate["White"]

        if total_games == 0:
            return 0

        if self.reward_by_player == 1.0:
            return (self.win_rate["Black"] / total_games) * 100
        else:
            return (self.win_rate["White"] / total_games) * 100
