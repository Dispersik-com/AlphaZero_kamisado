from collections import deque
import numpy as np
import json
from kamisado_environment.kamisado_enviroment import KamisadoGame
from kamisado_environment.pieces import Monk


class TreeSerializationMixin:
    def to_dict(self):
        """
        Convert the MCTS tree rooted at the given node to a dictionary using BFS.

        Args:
            node: The root node of the tree.

        Returns:
            A dictionary representation of the MCTS tree.
        """

        def array_to_str_list(array) -> list[str]:
            return [str(i) for i in array]

        if self.root.children is None:
            return None

        tree_data = {}
        queue = deque(self.root.children)

        while queue:
            current_node = queue.popleft()
            children_data = [child for child in current_node.children]
            tree_data[str(hash(current_node.state))] = {
                'state': array_to_str_list(current_node.state.game_board),
                'parent': str(hash(current_node.parent.state)),
                'action': current_node.action,
                'children': list(map(lambda x: str(hash(x.state)), children_data)),
                'visits': current_node.visits,
                'value': current_node.value,
                'untried_actions': str(current_node.untried_actions),
                'history_of_moves': array_to_str_list(current_node.state.history_of_moves),
            }

            queue.extend(current_node.children)

        return tree_data

    # def from_dict(self, dict_data):
    #     for _, data in dict_data.items():
    #         game = KamisadoEnvironment()
    #         game.game_board = self.str_list_to_game_board(data["state"])

    def save_tree(self, filename, indent=3):
        """
        Save the entire MCTS tree to a file using BFS-based serialization.

        Args:
            filename: The name of the file to save the tree.
        """
        tree_data = self.to_dict()

        with open(f"{filename}", 'w') as file:
            json.dump(tree_data, file, indent=indent)

    def load_tree(self, filename):
        """
        Load the MCTS tree from a file.

        Args:
            filename: The name of the file containing the tree.

        Returns:
            The dictionary representation of the loaded MCTS tree.
        """
        with open(filename, 'r') as file:
            tree_data = json.load(file)
        return tree_data

    @staticmethod
    def str_list_to_game_board(list_rows):
        converted_list = []

        for row in list_rows:
            for cell in row[1:-1].split(" "):
                if cell.isdigit():
                    converted_list.append(int(cell))
                else:
                    command = "Black" if cell[0] == "B" else "White"
                    converted_list.append(Monk(command, cell[2]))

        return np.array(converted_list).reshape((8, 8))
