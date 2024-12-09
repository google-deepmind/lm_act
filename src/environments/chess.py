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

"""Chess environment."""

import copy
import dataclasses
import io
import os
import pathlib
from typing import Literal

import chess
import chess.pgn
import dm_env
from numpy import random

from lm_act.src import bagz
from lm_act.src import config as config_lib
from lm_act.src import interfaces


ActionNotation = Literal['uci', 'san']


def action_to_string(
    action: chess.Move,
    board: chess.Board,
    action_notation: ActionNotation,
) -> str:
  """Returns the string representation of a chess action."""
  match action_notation:
    case 'uci':
      return str(action)
    case 'san':
      return board.san(action)


@dataclasses.dataclass(frozen=True, kw_only=True)
class EnvironmentConfig(config_lib.Environment):
  """Configuration for the environment."""

  name: str = 'chess'
  observation_type: Literal['fen', 'pgn', 'txt', 'png'] = 'fen'
  action_type: ActionNotation = 'san'

  stockfish_node_limit: int = 1
  stockfish_time_limit: float = 0.001
  stockfish_skill_level: int = 0


class Chess(interfaces.Environment):
  """A simple chess environment to play against stockfish."""

  def __init__(
      self,
      config: EnvironmentConfig,
      opening_paths: list[pathlib.Path] | None = None,
      openings: list[chess.pgn.Game] | None = None,
  ) -> None:
    self._stockfish_limit = chess.engine.Limit(
        nodes=config.stockfish_node_limit,
        time=config.stockfish_time_limit,
    )
    self._stockfish_skill_level = config.stockfish_skill_level
    self._action_notation = config.action_type

    if openings is not None:
      self._openings = openings
    elif opening_paths is not None:
      self._openings = list()
      for opening_path in opening_paths:
        pgns = bagz.BagReader(
            (opening_path / 'observations_pgn.bag').as_posix()
        )
        pgn = io.StringIO(pgns[0].decode('utf-8'))
        if (game := chess.pgn.read_game(pgn)) is not None:
          self._openings.append(game)
    else:
      raise ValueError('Either `openings` or `opening_paths` must be provided.')

    self._game: chess.pgn.Game = None
    self._node: chess.pgn.GameNode = None
    self._player_is_white: bool = None

  def reset(self) -> dm_env.TimeStep:
    self._game = self._openings.pop(0)
    self._node = self._game.end()
    self._player_is_white = self._board.turn == chess.WHITE
    return dm_env.restart(observation=self._observation)

  @property
  def _board(self):
    return self._node.board()

  @property
  def _observation(self):
    return {
        'board': copy.deepcopy(self._board),
        'fen': self._board.fen(en_passant='fen'),
        'pgn': str(self._game),
        'txt': str(self._board),
    }

  def _turn(
      self,
      action: str,
      notation: ActionNotation,
  ) -> None | dm_env.TimeStep:
    match notation:
      case 'uci':
        action = chess.Move.from_uci(action)
      case 'san':
        action = self._board.push_san(action)
    self._node = self._node.add_main_variation(action)

    if (
        claim_draw := self._board.can_claim_draw()
    ) or self._board.is_game_over():
      match (outcome := self._board.outcome(claim_draw=claim_draw)).winner:  # pytype: disable=attribute-error
        case chess.WHITE:
          reward = 1 if self._player_is_white else -1
        case chess.BLACK:
          reward = -1 if self._player_is_white else 1
        case None:
          reward = 0
        case _:
          raise ValueError(f'Unknown outcome: {outcome}')
      return dm_env.termination(observation=self._observation, reward=reward)

    return None

  def step(self, action: str) -> dm_env.TimeStep:
    outcome = self._turn(action=action, notation=self._action_notation)
    if outcome is not None:
      return outcome

    bin_path = os.path.join(
        os.getcwd(),
        '../Stockfish/src/stockfish',
    )
    with chess.engine.SimpleEngine.popen_uci(bin_path) as engine:
      engine.configure({'Skill Level': self._stockfish_skill_level})
      action = str(engine.play(self._board, limit=self._stockfish_limit).move)

    if (outcome := self._turn(action=action, notation='uci')) is not None:
      return outcome

    return dm_env.transition(observation=self._observation, reward=0)

  @property
  def legal_actions(self) -> list[str]:
    return sorted(
        action_to_string(
            action=action,
            board=self._board,
            action_notation=self._action_notation,
        )
        for action in self._board.legal_moves
    )

  def sample_legal_action(self, rng: random.Generator) -> str:
    return action_to_string(
        action=rng.choice(list(self._board.legal_moves)),
        board=self._board,
        action_notation=self._action_notation,
    )
