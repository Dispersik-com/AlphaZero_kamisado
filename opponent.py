from abc import ABC, abstractmethod


class Opponent(ABC):
    """
    Abstract base class for defining opponents in a game.

    Attributes:
        game: The game object representing the game being played.
        agent_player: The player object representing the agent player.
    """

    def __init__(self, game, agent_player):
        """
        Initialize the Opponent object.

        Args:
            game: The game object representing the game being played.
            agent_player: The player object representing the agent player.
        """
        self.agent_player = agent_player
        self.game = game
        self.current_node = None

    @abstractmethod
    def select_move(self, legal_actions):
        """
        Abstract method for selecting a move in the game.

        Args:
           legal_actions: List of legal actions available to the opponent.

        Returns:
           The selected move.
        """
        pass

    def set_current_node(self, node):
        self.current_node = node

    def get_current_node(self):
        return self.current_node


class MCTSOpponent(Opponent):
    """
    Opponent using Monte Carlo Tree Search (MCTS) algorithm for move selection.
    """

    def select_move(self, legal_actions):
        """
        Select a move using the MCTS algorithm.

        Args:
            legal_actions: List of legal actions available in the current game state.

        Returns:
            The selected move.
        """
        current_node = self.get_current_node()

        # Select a child node based on the chosen strategy
        best_child = current_node.select_child(strategy=self.agent_player.strategy)

        if best_child.action in legal_actions:
            return best_child.action
        else:
            raise ValueError("Selected action is not legal.")
