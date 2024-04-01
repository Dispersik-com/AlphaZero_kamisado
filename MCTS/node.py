import math
import random


class MCTSNode:
    def __init__(self, state, parent=None, action=None, is_root=False):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = self.get_legal_actions()
        self.is_root = is_root

    def get_legal_actions(self):
        """
        Get the legal actions available in the current state.

        Returns:
            A list of tuples representing legal actions.
        """
        return self.state.get_legal_moves()

    def is_fully_expanded(self):
        """
        Check if all possible actions have corresponding child nodes.

        Returns:
            True if fully expanded, False otherwise.
        """
        return not bool(self.untried_actions)

    def select_child(self, exploration_weight=1.0, strategy="UCB1", opponent=False):
        """
        Select a child node based on a selection strategy (e.g., UCB1).

        Args:
            exploration_weight: The weight for exploration in the selection strategy.
            strategy: The selection strategy to use.
            opponent: Flag indicating whether the node is being selected by the opponent.

        Returns:
            The selected child node.
            :param exploration_weight:
            :param strategy:
        """
        filtered_children = list(filter(lambda child: child.visits != 0, self.children))

        if not filtered_children or not self.children:
            random_move = random.choice(self.get_legal_actions())
            return self.expand(random_move)

        if strategy == "UCB1":
            selection_function = lambda child: child.value / child.visits + exploration_weight * math.sqrt(
                math.log(self.visits) / child.visits)

        elif strategy == "UCB1-Tuned":
            selection_function = lambda child: child.value / child.visits + exploration_weight * math.sqrt(
                math.log(self.parent.visits) / (2 * child.visits))

        elif strategy == "Exp3":
            selection_function = lambda child: ((1 - exploration_weight) * child.value / child.visits +
                                                exploration_weight / len(self.children))

        elif strategy == "ThompsonSampling":
            selection_function = lambda child: random.betavariate(child.value + 1, child.visits - child.value + 1)

        else:
            raise ValueError("Invalid strategy specified")

        if opponent:
            return min(filtered_children, key=selection_function)

        selected_child = max(filtered_children, key=selection_function)

        return selected_child

    def expand(self, action=None):
        """
        Expand the node by adding a child with an untried action.

        Returns:
            The created child node.
        """
        if action is None:
            # If no specific action is provided for expanding the node:

            # Check if the node is fully expanded (all possible actions are available)
            if self.is_fully_expanded():
                # selected_child = random.choice(self.children)
                selected_child = self.select_child()
                return selected_child

            # Get a list of legal actions that can be applied to the current node
            legal_moves_by_piece = self.get_legal_actions()

            # Choose one of the untried actions that has not been used in this node yet
            action = list(set(self.untried_actions) & set(legal_moves_by_piece)).pop()
            # Remove the chosen action from the list of untried actions
            self.untried_actions.pop(self.untried_actions.index(action))

        # Perform the action in a copy of the current state to create a new state
        new_state = self.state.copy_and_apply_action(action)
        child_node = MCTSNode(new_state, parent=self, action=action)

        # Check for duplicates among the child nodes by states
        new_state = child_node.state
        if any(hash(child.state) == hash(new_state) for child in self.children):
            return child_node

        self.children.append(child_node)
        return child_node

    def is_terminal(self):
        """
        Check if the current state is a terminal state.

        Returns:
            True if terminal, False otherwise.
        """
        self.state.check_winner()
        has_action = bool(len(self.get_legal_actions()) != 0)

        return self.state.winner is not None and has_action

    def is_winner(self, player):
        """
        Check if the current state is a winning state for the specified player.

        Args:
            player: The player for whom to check the winning state.

        Returns:
            True if winning, False otherwise.
        """
        self.state.check_winner()

        return self.state.winner == player
