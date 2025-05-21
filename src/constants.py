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

"""Constants used in the LMAct project."""

from typing import TypeVar

from lm_act.src import interfaces
from lm_act.src.agents import chess as chess_agents
from lm_act.src.agents import crossword as crossword_agents
from lm_act.src.agents import grid_world as grid_world_agents
from lm_act.src.agents import random as random_agents
from lm_act.src.agents import tic_tac_toe as tic_tac_toe_agents
from lm_act.src.environments import chess as chess_env
from lm_act.src.environments import crossword as crossword_env
from lm_act.src.environments import dm_control as dm_control_env
from lm_act.src.environments import grid_world as grid_world_env
from lm_act.src.environments import tic_tac_toe as tic_tac_toe_env


def get_rgb_shape(environment_name: str) -> tuple[int, int, int]:
  if environment_name.startswith('atari'):
    return (210, 160, 3)
  elif environment_name == 'chess':
    return (256, 256, 3)
  elif environment_name.startswith('tic_tac_toe'):
    return (256, 256, 3)
  elif environment_name.startswith('grid_world'):
    return (192, 192, 3)
  else:
    raise ValueError(f'Unknown environment name: {environment_name}.')


# The interfaces are abstract, so we need to use TypeVar to make them generic.
Agent = TypeVar('Agent', bound=interfaces.Agent)
Environment = TypeVar('Environment', bound=interfaces.Environment)


def get_agent_builder(agent_name: str) -> type[Agent]:
  match agent_name:
    case 'random':
      return random_agents.RandomAgent
    case 'chess_stockfish':
      return chess_agents.StockfishAgent
    case 'crossword_oracle':
      return crossword_agents.OracleAgent
    case 'grid_world_shortest_path':
      return grid_world_agents.ShortestPathAgent
    case 'tic_tac_toe_minimax':
      return tic_tac_toe_agents.MinimaxAgent
    case _:
      raise ValueError(f'Unknown agent name: {agent_name}.')


def get_environment_builder(environment_name: str) -> type[Environment]:
  """Returns the environment builder for the given environment name."""
  if environment_name.startswith('atari'):
    raise NotImplementedError('atari environments are not yet supported.')
  elif environment_name == 'chess':
    return chess_env.Chess
  elif environment_name == 'crossword':
    return crossword_env.Crossword
  elif environment_name.startswith('dm_control'):
    return dm_control_env.DMControl
  elif environment_name.startswith('grid_world'):
    return grid_world_env.GridWorld
  elif environment_name.startswith('tic_tac_toe'):
    return tic_tac_toe_env.TicTacToe
  else:
    raise ValueError(f'Unknown environment name: {environment_name}.')
