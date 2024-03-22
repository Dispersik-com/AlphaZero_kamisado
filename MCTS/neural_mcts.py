import random
from collections import deque
from MCTS.mcts import MonteCarloTreeSearch
import torch


class NeuralMonteCarloTreeSearch(MonteCarloTreeSearch):
    def __init__(self, game, policy_network, value_network, player="Black",  update_form_buffer=True, batch_size=32):
        super().__init__(game, player=player)
        # add policy and value networks
        self.policy_network = policy_network
        self.value_network = value_network

        self.opponent_player = None

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

    def set_opponent(self, opponent):
        self.opponent_player = opponent

    def neural_network_select(self, input_data, legal_actions):

        # get move probabilities from the policy_network
        action_probs = self.policy_network(input_data)

        # apply mask by legal moves
        mask = self.policy_network.create_mask(legal_actions)
        action_probs_by_legal_moves = action_probs * mask.float()

        # choose a move according to the probabilities
        action_index = torch.argmax(action_probs_by_legal_moves).item()
        selected_action = self.policy_network.action_labels[action_index]
        return selected_action

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

            if self.opponent_player is not None and self.player != current_node.state.current_player:
                self.opponent_player.set_current_node(current_node)
                selected_action = self.opponent_player.select_move(legal_actions)
            else:
                selected_action = self.neural_network_select(input_data, legal_actions)

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

        self.value_network.train()
        self.policy_network.train()

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

    def validate_searching(self, num_validations=100, validate_strategy="UCB1"):
        """
        Method to validate the searching process of the MCTS algorithm.

        Args:
            validate_strategy (str): Strategy used for validation. Default is "UCB1".
                Available strategies: "UCB1", "UCB1-Tuned", "Epx3", "ThompsonSampling".
            num_validations (int): Number of validation runs to perform.

        Returns:
            dict: Validation data containing network predictions and target data.
                - "value_estimations": List of estimated values by the value network.
                - "policy_estimations": List of estimated policies by the policy network.
                - "true_values": List of true values from the current nodes.
                - "expert_moves": List of expert moves selected during validation.

        Note:
            This method evaluates the policy and value networks using the provided number of validations.
            It simulates MCTS searches and collects data for validation purposes.
        """
        self.value_network.eval()
        self.policy_network.eval()

        validation_data = {
            # networks predictions
            "value_estimations": [],
            "policy_estimations": [],

            # target data
            "true_values": [],
            "expert_moves": [],
        }

        for i in range(num_validations):

            root = self.root
            queue = deque([root])

            current_node = self.root

            # Collecting data for model validation
            while queue and current_node.get_legal_actions():

                input_data = self.convert_node_to_input(current_node)

                with torch.no_grad():
                    output_value = self.value_network(input_data.detach().clone())
                    output_policy = self.policy_network(input_data.detach().clone())

                    validation_data["value_estimations"].append(output_value.item())
                    validation_data["policy_estimations"].append(output_policy.detach().clone())

                node_value = torch.tanh(torch.tensor(current_node.value))
                validation_data["true_values"].append(torch.tanh(node_value).item())

                expert_action = current_node.select_child(strategy=validate_strategy).action
                validation_data["expert_moves"].append(expert_action)

                queue.extend([random.choice(current_node.children)])

                current_node = queue.popleft()

        return validation_data

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
