# Copyright 2025 Google LLC
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

"""Grid World environment."""

from collections.abc import Mapping
import copy
import dataclasses
import os
import pathlib
from typing import Literal

import dm_env
import imageio
import jax
import numpy as np

from lm_act.src import bagz
from lm_act.src import config as config_lib
from lm_act.src import interfaces


_ASSETS_PATH = pathlib.Path(
    os.path.join(os.getcwd(), '../crafter/crafter/assets')
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class EnvironmentConfig(config_lib.Environment):
  """Configuration for the environment."""

  name: str = 'grid_world'
  observation_type: Literal['rgb', 'txt', 'coords'] = 'rgb'
  height: int = 12
  width: int = 12

  def __post_init__(self):
    object.__setattr__(self, 'name', f'grid_world_{self.height}x{self.width}')


class GridWorld(interfaces.Environment):
  """2D grid world environment with a player and a target."""

  def __init__(
      self,
      config: EnvironmentConfig,
      opening_paths: list[pathlib.Path] | None = None,
      openings: list[tuple[np.ndarray, np.ndarray]] | None = None,
  ) -> None:
    if openings is not None:
      self._openings = openings
    elif opening_paths is not None:
      self._openings = list()
      for opening_path in opening_paths:
        player_coordinates = bagz.BagReader(
            (opening_path / 'observations_player.bag').as_posix()
        )
        target_coordinates = bagz.BagReader(
            (opening_path / 'observations_target.bag').as_posix()
        )
        player = np.frombuffer(player_coordinates[0], dtype=np.int64)
        target = np.frombuffer(target_coordinates[0], dtype=np.int64)
        self._openings.append((player, target))
    else:
      raise ValueError('Either `openings` or `opening_paths` must be provided.')

    self._width = config.width
    self._height = config.height

    self._player: np.ndarray = None
    self._target: np.ndarray = None

    self._walls = np.full((self._height, self._width), False, dtype=bool)
    self._walls[0, :] = True
    self._walls[-1, :] = True
    self._walls[:, 0] = True
    self._walls[:, -1] = True

    with open(_ASSETS_PATH / 'food.png', 'r') as f:
      target_sprite = imageio.imread(f)[:, :, :-1]
    with open(_ASSETS_PATH / 'player.png', 'r') as f:
      player_sprite = imageio.imread(f)[:, :, :-1]
    with open(_ASSETS_PATH / 'stone.png', 'r') as f:
      wall_sprite = imageio.imread(f)

    sprite_matrix = np.array([wall_sprite, target_sprite, player_sprite])
    self._sprite_matrix = np.transpose(sprite_matrix[:, ::-1], [1, 2, 0, 3])

    self._rgb = jax.jit(self._rgb)

  def reset(self) -> dm_env.TimeStep:
    self._player, self._target = self._openings.pop(0)
    return dm_env.restart(observation=self._observation)

  def _rgb(self, state: np.ndarray) -> jax.Array:
    return jax.lax.conv_transpose(
        state[None],  # NHWC
        self._sprite_matrix,  # HWIO
        (16, 16),
        'SAME',
    )[0]

  @property
  def _text(self) -> str:
    scene = list()

    for row in range(self._height):
      row_str = '|'
      for col in range(self._width):
        if self._walls[row, col]:
          tile = 'wall'
        elif self._player[0] == row and self._player[1] == col:
          tile = 'player'
        elif self._target[0] == row and self._target[1] == col:
          tile = 'target'
        else:
          tile = 'tile'
        row_str += tile.center(8) + '|'
      scene.append(row_str)
      scene.append('-' * len(row_str))

    scene = ['-' * len(scene[0])] + scene

    return '\n'.join(scene)

  @property
  def _observation(self) -> Mapping[str, np.ndarray | str]:
    player_state = np.zeros((self._height, self._width), dtype=np.bool_)
    player_state[self._player[0], self._player[1]] = True
    target_state = np.zeros((self._height, self._width), dtype=np.bool_)
    target_state[self._target[0], self._target[1]] = True
    state = np.stack(
        [self._walls, target_state, player_state],
        axis=-1,
        dtype=np.uint8,
    )
    return {
        'player': copy.deepcopy(self._player),
        'target': copy.deepcopy(self._target),
        'rgb': copy.deepcopy(np.array(self._rgb(state), dtype=np.uint8)),
        'txt': self._text,
        'coords': str(
            dict(player=self._player.tolist(), target=self._target.tolist())
        ),
    }

  def step(self, action: str) -> dm_env.TimeStep:
    next_y, next_x = self._player

    match action:
      case 'left':
        next_x -= 1
      case 'right':
        next_x += 1
      case 'up':
        next_y -= 1
      case 'down':
        next_y += 1
      case _:
        raise ValueError(f'Unsupported action: {action}')

    if not self._walls[next_y, next_x]:
      self._player = np.array([next_y, next_x])
      if np.all(self._player == self._target):
        return dm_env.termination(observation=self._observation, reward=1)

    return dm_env.transition(observation=self._observation, reward=0)

  @property
  def legal_actions(self) -> list[str]:
    """Returns the legal actions."""
    return ['left', 'right', 'up', 'down']
