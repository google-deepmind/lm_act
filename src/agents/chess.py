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

"""Random and Stockfish agents for chess."""

import dataclasses
import os
from typing import Any

import chess
import chess.engine
import chess.pgn
import chess.svg
import numpy as np

from lm_act.src import config as config_lib
from lm_act.src import interfaces
from lm_act.src.environments import chess as chess_env


@dataclasses.dataclass(frozen=True, kw_only=True)
class StockfishAgentConfig(config_lib.Agent):
  """Configuration for the Stockfish agent."""

  name: str = 'chess_stockfish'
  action_type: chess_env.ActionNotation = 'san'
  skill_level: int = 20
  time_limit: float = 0.05
  node_limit: int | None = None


class StockfishAgent(interfaces.Agent):
  """Stockfish agent."""

  def __init__(
      self,
      config: StockfishAgentConfig,
  ) -> None:
    self._skill_level = config.skill_level
    self._action_notation = config.action_type
    self._limit = chess.engine.Limit(
        nodes=config.node_limit,
        time=config.time_limit,
    )

  def step(
      self,
      observation: Any,
      environment: interfaces.Environment,
      rng: np.random.Generator,
  ) -> str:
    """Returns Stockfish's action for the board in the observation."""
    bin_path = os.path.join(
        os.getcwd(),
        '../Stockfish/src/stockfish',
    )
    with chess.engine.SimpleEngine.popen_uci(bin_path) as engine:
      engine.configure({'Skill Level': self._skill_level})
      return chess_env.action_to_string(
          action_notation=self._action_notation,
          action=engine.play(observation['board'], limit=self._limit).move,
          board=observation['board'],
      )
