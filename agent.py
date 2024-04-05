import config
from MCTS.neural_mcts import NeuralMonteCarloTreeSearch
from policy_value_networks import PolicyNet, ValueNet


class Agent(NeuralMonteCarloTreeSearch):
    """
    Class representing an agent using neural networks to make decisions in a game.
    Inherits functionality from the NeuralMonteCarloTreeSearch class.

    Attributes:
        game: The game object in which the agent participates.
        player: The player for which the agent plays. Default is "White".
    """

    def __init__(self, game, player="White"):
        """
        Initializes the agent with the given game and player.

        Args:
            game: The game object in which the agent participates.
            player: The player for which the agent plays. Default is "White".
        """
        # Creating policy and value neural networks with specified learning rates
        policy_net = PolicyNet(learning_rate=config.policy_learning_rate)
        value_net = ValueNet(learning_rate=config.value_learning_rate)
        # Initializing the agent using the parent constructor
        super().__init__(game=game,
                         player=player,
                         policy_network=policy_net,
                         value_network=value_net,
                         update_form_buffer=config.batch_learning,
                         batch_size=config.batch_size)

    def actions_from_probs(self, net_probs):
        action_labels = self.policy_network.action_labels
        actions_list = []
        for i, prob_move in enumerate(net_probs.tolist()[0]):
            if prob_move > 0:
                actions_list.append(action_labels[i])

        return actions_list

    def load_models(self, policy_file, value_file):
        """
        Loads trained models for the policy and value neural networks from files.

        Args:
            policy_file: File containing the trained model for the policy neural network.
            value_file: File containing the trained model for the value neural network.
        """
        self.policy_network.load_model(policy_file)
        self.value_network.load_model(value_file)

    def find_best_move(self, game, next_move=None, max_depth=3, max_width=5):
        """
        Finds the best move for the current game state using recursive search.

        Args:
            next_move:
            game: The game object representing the current state of the game.
            max_depth: Maximum recursion depth for move search. Default is 3.
            max_width: Maximum number of moves to consider at each recursion level. Default is 5.

        Returns:
            The best found move.
        """
        state = game.game_board
        input_data = self.convert_state_to_input(state)
        game.check_winner()

        # If maximum depth is reached or there's a winner, return the state evaluation
        if max_depth == 0 or game.winner is not None:
            return next_move, self.value_network(input_data)

        # Get action probabilities from the neural network
        actions_probs = self.neural_network_select(input_data, game.get_legal_moves(), get_all_probs=True)

        # Filter out zero values and sort action probabilities in descending order
        available_moves = self.actions_from_probs(actions_probs)
        available_moves = available_moves[:max_width]

        # Traverse moves and select the best one
        best_move = None
        best_score = float('-inf')

        for move in available_moves:
            copy_game = game.copy_and_apply_action(move)
            next_legal_action = copy_game.get_legal_moves()

            # If no legal actions or move is a pass, move to the next move
            if not next_legal_action:
                if not copy_game.pass_move():
                    break
                continue

            # Recursively find the best move for the next game state
            _, score = self.find_best_move(copy_game, next_move=move,
                                           max_depth=max_depth - 1,
                                           max_width=max_width)

            # Update the best move if the current move yields a better result
            if score > best_score:
                best_score = score
                best_move = move

        return best_move, best_score

    def update_game(self, game):
        """
        Updates the agent's game state.

        Args:
            game: The new game state.
        """
        self.game = game
