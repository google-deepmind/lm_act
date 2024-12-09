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

"""Tic-Tac-Toe environment."""

import copy
import dataclasses
import io
import pathlib
from typing import Literal

import dm_env
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw

from lm_act.src import bagz
from lm_act.src import config as config_lib
from lm_act.src import interfaces


ROWS = ['a', 'b', 'c']
COLS = ['1', '2', '3']
AXES = [
    # Rows.
    ('row_0', np.array([0, 1, 2])),
    ('row_1', np.array([3, 4, 5])),
    ('row_2', np.array([6, 7, 8])),
    # Columns.
    ('col_0', np.array([0, 3, 6])),
    ('col_1', np.array([1, 4, 7])),
    ('col_2', np.array([2, 5, 8])),
    # Negative diagonal.
    ('neg_diag', np.array([0, 4, 8])),
    # Positive diagonal.
    ('pos_diag', np.array([6, 4, 2])),
]


def _draw_board(
    board: np.ndarray,
    size: int = 768,
    color: str = 'black',
    background: str = 'white',
    render_coordinates: bool = True,
) -> tuple[bytes, np.ndarray]:
  """Returns the board as a PNG image and RGB array."""
  image = Image.new('RGB', size=(size, size), color=background)
  draw = ImageDraw.Draw(image)
  width = size // 25

  # Draw the grid.
  draw.line(
      ((size // 3, 0), (size // 3, size)),
      fill=color,
      width=width,
  )
  draw.line(
      ((2 * size // 3, 0), (2 * size // 3, size)),
      fill=color,
      width=width,
  )
  draw.line(
      ((0, size // 3), (size, size // 3)),
      fill=color,
      width=width,
  )
  draw.line(
      ((0, 2 * size // 3), (size, 2 * size // 3)),
      fill=color,
      width=width,
  )

  # Draw the symbols.
  for row_idx, row in enumerate(board):
    for col_idx, symbol in enumerate(row):

      def _to_coord(idx: int, offset: int) -> int:
        return (4 * idx + offset) * size // 12

      coords = {
          'top_left': (_to_coord(col_idx, 1), _to_coord(row_idx, 1)),
          'top_right': (_to_coord(col_idx, 1), _to_coord(row_idx, 3)),
          'bottom_left': (_to_coord(col_idx, 3), _to_coord(row_idx, 1)),
          'bottom_right': (_to_coord(col_idx, 3), _to_coord(row_idx, 3)),
      }
      match symbol:
        case 'o':
          draw.ellipse(
              [coords['top_left'], coords['bottom_right']],
              outline=color,
              width=width,
          )
        case 'x':
          draw.line(
              (coords['top_left'], coords['bottom_right']),
              fill=color,
              width=width,
          )
          draw.line(
              (coords['top_right'], coords['bottom_left']),
              fill=color,
              width=width,
          )

  if render_coordinates:
    image = Image.fromarray(_add_coordinates(np.array(image)))

  with io.BytesIO() as buffer:
    image.save(buffer, format='PNG')
    return buffer.getvalue(), np.array(image)


def _add_coordinates(rgb_image: np.ndarray) -> np.ndarray:
  """Adds coordinates to the image.

  Args:
    rgb_image: The RGB image array to add coordinates to.

  Returns:
    The RGB image array with coordinates added.
  """
  rgb_image = rgb_image.astype(np.uint8)
  height, width, _ = rgb_image.shape

  x_ticks = np.linspace(height / 6, 5 * height / 6, 3)
  y_ticks = np.linspace(width / 6, 5 * width / 6, 3)

  x_tick_labels = ['1', '2', '3']
  y_tick_labels = ['a', 'b', 'c']

  font_size = round(8 / 256 * height)

  fig = plt.figure(figsize=(height / 100, width / 100))
  plt.imshow(rgb_image)

  plt.xticks(x_ticks, x_tick_labels, fontsize=font_size)
  plt.yticks(y_ticks, y_tick_labels, fontsize=font_size)

  plt.tick_params(
      axis='both',
      which='both',
      labeltop=True,
      labelright=True,
      labelbottom=True,
      labelleft=True,
      length=0,
  )

  fig.tight_layout()
  fig.canvas.draw()
  new_width, new_height = fig.canvas.get_width_height()
  rgb_buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

  plt.close(fig)

  return rgb_buffer.reshape((new_height, new_width, 3))


def legal_actions(board: np.ndarray) -> list[str]:
  return [
      ROWS[coords[0]] + COLS[coords[1]] for coords in np.argwhere(board == ' ')
  ]


@dataclasses.dataclass(frozen=True, kw_only=True)
class EnvironmentConfig(config_lib.Environment):
  """Configuration for the environment."""

  name: str = 'tic_tac_toe'
  observation_type: Literal['txt', 'png'] = 'png'
  render_coordinates: bool = False
  seed: int = 0

  def __post_init__(self):
    if self.render_coordinates:
      if self.observation_type == 'txt':
        raise ValueError(
            'Rendering coordinates is only supported for `png` observations.'
        )
      object.__setattr__(self, 'name', 'tic_tac_toe_with_coordinates')


class TicTacToe(interfaces.Environment):
  """A simple tic-tac-toe environment to play against a random policy."""

  def __init__(
      self,
      config: EnvironmentConfig,
      opening_paths: list[pathlib.Path] | None = None,
      openings: list[tuple[np.ndarray, bool]] | None = None,
  ) -> None:
    if openings is not None:
      self._openings = openings
    elif opening_paths is not None:
      self._openings = list()
      for opening_path in opening_paths:
        boards = bagz.BagReader(
            (opening_path / 'observations_board.bag').as_posix()
        )
        board = np.frombuffer(boards[0], dtype=np.dtype('<U1')).reshape((3, 3))
        player_is_x = np.sum(board != ' ') % 2 == 0
        self._openings.append((board, player_is_x))
    else:
      raise ValueError('Either `openings` or `opening_paths` must be provided.')

    self._rng = np.random.default_rng(seed=config.seed)
    self._board: np.ndarray = None
    self._player_is_x: bool = None
    self._render_coordinates: bool = config.render_coordinates

  def reset(self) -> dm_env.TimeStep:
    self._board, self._player_is_x = self._openings.pop(0)
    self._board = copy.deepcopy(self._board)
    return dm_env.restart(observation=self._observation)

  @property
  def _observation(self):
    return {
        'board': copy.deepcopy(self._board),
        'txt': '\n----------\n'.join(' | '.join(row) for row in self._board),
        # Gemini resizes all images to 768x768, so we might as well do it here.
        'png': _draw_board(
            board=self._board,
            size=768,
            render_coordinates=self._render_coordinates,
        )[0],
        # In contrast, Sequence Storage can only render images up to 256x256.
        'rgb': _draw_board(
            board=self._board,
            size=256,
            render_coordinates=self._render_coordinates,
        )[1],
        'symbol': self.symbol(is_player=True),
    }

  def symbol(self, is_player: bool) -> str:
    if is_player:
      return 'x' if self._player_is_x else 'o'
    return 'o' if self._player_is_x else 'x'

  def _turn(
      self,
      action: str,
      is_player: bool,
  ) -> None | dm_env.TimeStep:
    symbol = self.symbol(is_player=is_player)
    self._board[ROWS.index(action[0]), COLS.index(action[1])] = symbol

    for _, axis in AXES:
      line = np.take_along_axis(self._board.flatten(), axis, axis=None)
      if (line == symbol).all():
        return dm_env.termination(
            observation=self._observation,
            reward=1 if is_player else -1,
        )

    if ' ' not in self._board:
      return dm_env.termination(observation=self._observation, reward=0)

    return None

  def step(self, action: str) -> dm_env.TimeStep:
    if action not in self.legal_actions:
      raise ValueError(f'Action {action} is illegal.')

    if (outcome := self._turn(action=action, is_player=True)) is not None:
      return outcome

    # The adversary randomly chooses a legal action.
    move = self._rng.choice(self.legal_actions)

    if (outcome := self._turn(action=move, is_player=False)) is not None:
      return outcome

    return dm_env.transition(observation=self._observation, reward=0)

  @property
  def legal_actions(self) -> list[str]:
    return legal_actions(self._board)
