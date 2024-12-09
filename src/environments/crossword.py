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

"""Crossword environment."""

import ast
import collections
from collections.abc import Mapping
import dataclasses
import logging
import os
import pathlib
import re
from typing import Literal

import dm_env

from lm_act.src import bagz
from lm_act.src import config as config_lib
from lm_act.src import interfaces


_BASE_DIR_PATH = pathlib.Path(
    os.path.join(
        os.getcwd(),
        'data/lm_act/',
    )
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class EnvironmentConfig(config_lib.Environment):
  """Configuration for the environment."""

  name: str = 'crossword'
  observation_type: Literal['txt'] = 'txt'


class Crossword(interfaces.Environment):
  """A simple crossword environment."""

  def __init__(
      self,
      config: EnvironmentConfig,
      opening_paths: list[pathlib.Path] | None = None,
      openings: list[str] | None = None,
  ) -> None:
    if openings is not None:
      self._openings = openings
    elif opening_paths is not None:
      self._openings = list()
      for opening_path in opening_paths:
        puzzles = bagz.BagReader(
            (opening_path / 'observations_puzzle.bag').as_posix()
        )
        self._openings.append(puzzles[0].decode('utf-8'))
    else:
      raise ValueError('Either `openings` or `opening_paths` must be provided.')

    self._puzzle: str = None
    self._grid: str = None
    self._clues: str = None
    self._coords_by_solution: Mapping[str, tuple[int, int]] = None
    self._unfound_solutions: list[str] = None

    self._words_by_length = collections.defaultdict(set)

    with open(_BASE_DIR_PATH / 'crossword' / 'words.txt', 'r') as f:
      for word in f:
        word = word.strip()
        self._words_by_length[len(word)].add(word)

  def reset(self) -> dm_env.TimeStep:
    self._puzzle = self._openings.pop(0)
    self._grid, self._clues, solutions, *_ = self._puzzle.split('\n\n')

    if (grid_len := len(self._grid)) != 554:
      raise ValueError(f'Invalid grid size, should be 554 but is {grid_len}.')

    self._coords_by_solution = dict()
    for solution in solutions.split('Solution:\n')[1].split('\n'):
      # The solution has the format "A1: word (row, col)", so we need some regex
      # magic to separate the word and the coordinates.
      if (match := re.match(r'(.*?)\s*(\(\d+,\s*\d+\))', solution)) is not None:
        sol, coords = match.groups()
        self._coords_by_solution[sol] = ast.literal_eval(coords)

    self._unfound_solutions = list(self._coords_by_solution.keys())

    return dm_env.restart(observation=self._observation)

  @property
  def _observation(self):
    return {
        'txt': self._grid + '\n\n' + self._clues,
        'puzzle': (
            self._grid
            + '\n\n'
            + self._clues
            + '\n\nSolution:\n'
            + '\n'.join(
                f'{sol} {self._coords_by_solution[sol]}'
                for sol in self._unfound_solutions
            )
        ),
    }

  def _update_grid(self, solution: str) -> None:
    is_vertical = solution[0] == 'D'
    row, col = self._coords_by_solution[solution]

    # Account for the offsets.
    row = row * 2 + 1
    col = col * 5 + 3

    for character in solution.split(': ')[1]:
      idx = row * 37 + col
      self._grid = (
          self._grid[: idx - 1] + ' ' + character + self._grid[idx + 1 :]
      )
      row += 2 * is_vertical
      col += 5 * (1 - is_vertical)

  def step(self, action: str) -> dm_env.TimeStep:
    # The environment should not be case-sensitive.
    action = action.upper().strip()

    if action in self._unfound_solutions:
      self._unfound_solutions.remove(action)
      self._update_grid(action)

      if self._unfound_solutions:
        return dm_env.transition(reward=1, observation=self._observation)
      return dm_env.termination(reward=1, observation=self._observation)

    if action in self._coords_by_solution.keys():
      return dm_env.transition(reward=0, observation=self._observation)

    # We continue if the proposed word is incorrect but has the correct length.
    try:
      idx, word = action.split(': ')
      solution = [
          sol for sol in self._coords_by_solution if sol.startswith(idx)
      ][0]
      if len(solution.split(': ')[1]) == len(word):
        return dm_env.transition(reward=0, observation=self._observation)
    except (IndexError, ValueError):
      logging.info('Invalid action: %s', action)
      return dm_env.termination(reward=-1, observation=self._observation)

    # Otherwise, we terminate the game.
    return dm_env.termination(reward=-1, observation=self._observation)

  def action_is_invalid(self, action: str) -> bool:
    """Returns whether the action in the format `A0: word` or `D1: word`."""
    return not re.match(r'^[AD]\d+:\s\w+$', action)

  @property
  def legal_actions(self) -> list[str]:
    """Returns the legal actions (all possible words of the correct length)."""
    legal_actions = list()
    for solution in self._unfound_solutions:
      idx, fill = solution.split(': ')
      for word in self._words_by_length[len(fill)]:
        legal_actions.append(f'{idx}: {word}')
    return legal_actions
