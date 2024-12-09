# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Minimax agent for tic tac toe."""

import copy
import dataclasses
from typing import Any

import numpy as np

from lm_act.src import config as config_lib
from lm_act.src import interfaces
from lm_act.src.environments import tic_tac_toe as tic_tac_toe_env


@dataclasses.dataclass(frozen=True, kw_only=True)
class MinimaxAgentConfig(config_lib.Agent):
  """Configuration for the minimax agent."""

  name: str = 'tic_tac_toe_minimax'


def _minimax(
    player_symbol: str,  # Constant across the recursion.
    board: np.ndarray,
    current_symbol: str,
    depth: int,
    is_max: bool,
) -> tuple[str | None, int]:
  """Returns the best move and its value for the given board and symbol."""
  # Check whether there is a winner.
  for _, axis in tic_tac_toe_env.AXES:
    line = np.take_along_axis(board.flatten(), axis, axis=None)
    if (line == player_symbol).all():
      return None, 10 - depth
    elif (line == ('o' if player_symbol == 'x' else 'x')).all():
      return None, -10 + depth

  # Check whether there is a draw.
  if ' ' not in board:
    return None, 0

  # Search all the legal moves.
  best_value = -100 if is_max else 100
  best_move = None

  for move in tic_tac_toe_env.legal_actions(board):
    row = tic_tac_toe_env.ROWS.index(move[0])
    col = tic_tac_toe_env.COLS.index(move[1])
    new_board = copy.deepcopy(board)
    new_board[row, col] = current_symbol
    _, value = _minimax(
        player_symbol=player_symbol,
        board=new_board,
        current_symbol='o' if current_symbol == 'x' else 'x',
        is_max=not is_max,
        depth=depth + 1,
    )

    if (is_max and best_value < value) or (not is_max and value < best_value):
      best_value = value
      best_move = move

  return best_move, best_value


class MinimaxAgent(interfaces.Agent):
  """Minimax agent for tic tac toe."""

  def __init__(self, config: MinimaxAgentConfig):
    pass

  def step(
      self,
      observation: Any,
      environment: interfaces.Environment,
      rng: np.random.Generator,
  ) -> str:
    """Returns the best move for the given board."""
    action, _ = _minimax(
        player_symbol=observation['symbol'],
        board=observation['board'],
        current_symbol=observation['symbol'],
        depth=0,
        is_max=True,
    )
    if action is None:
      raise ValueError('No optimal action found.')
    return action
