# AlphaZero_kamisado

### General Description:
The *"AlphaZero Kamisado"* project is an implementation of the AlphaZero algorithm for playing Kamisado. AlphaZero is a reinforcement learning algorithm developed by **DeepMind** that achieves self-learning without expert-labeled data, using only knowledge of the environment. Kamisado is a strategic tabletop game for two players, where each player controls eight pieces of a specific color, which move on a multi-colored board. The color of the square on which a player moves determines the color of the piece that the next player must move.

**Project Structure**:

- **game_environment**:
  - **kamisado_board.py**: Implementation of the game board.
  - **kamisado_enviroment.py**: Implementation of the Kamisado game environment.
  - **kamisado_move_validator.py**: Move validator for playing Kamisado.
  - **pieces.py**: Classes for game pieces.
- **kamisado_web_game**:
  - **web_game.html**: HTML page with the game interface.
  - **styles.css**: CSS file for styling the interface.
  - **script.js**: JavaScript script for interacting with the game.
  - **web_server.py**: Web server for handling requests from players. Web server based on aiohttp.
- **MCTS**:
  - **mcts.py**: Implementation of classical Monte Carlo for playing Kamisado.
  - **neural_mcts.py**: Implementation of Monte Carlo with a neural network for playing Kamisado.
  - **node.py**: Tree node for Monte Carlo.
  - **tree_serializer.py**: Module for serialization and deserialization of the Monte Carlo tree.
- **opponent.py**:
  - **Opponent.py**: Abstract opponent class.
    - *MCTSOpponent*: Opponent using Monte Carlo for move selection.
- **policy_value_networks.py**:
  - *PolicyNet*: Neural network for policy evaluation.
  - *ValueNet*: Neural network for state value evaluation in the game.
- **metrics.py**: Module with functions for evaluating model training quality.
- **net_utils.py**: Auxiliary module with an interface for saving and loading models.
- **AlphaZero_train.py**: The main script for training the AlphaZero model for playing Kamisado.
- **config.py**: Module with configuration parameters for training and running the project.






