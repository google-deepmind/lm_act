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

"""Shortest path agent for grid world."""

from collections.abc import Mapping
import dataclasses
from typing import Any

import numpy as np

from lm_act.src import config as config_lib
from lm_act.src import interfaces


@dataclasses.dataclass(frozen=True, kw_only=True)
class ShortestPathAgentConfig(config_lib.Agent):
  """Configuration for the minimax agent."""

  name: str = 'grid_world_shortest_path'


class ShortestPathAgent(interfaces.Agent):
  """Shortest path agent for grid world."""

  def __init__(self, config: ShortestPathAgentConfig) -> None:
    pass

  def step(
      self,
      observation: Mapping[str, Any],
      environment: interfaces.Environment,
      rng: np.random.Generator,
  ) -> str:
    """Returns an optimal action for the observation and legal actions."""
    player_y, player_x = observation['player']
    target_y, target_x = observation['target']

    optimal_actions = list()

    if target_x < player_x:
      optimal_actions.append('left')
    elif player_x < target_x:
      optimal_actions.append('right')
    if target_y < player_y:
      optimal_actions.append('up')
    elif player_y < target_y:
      optimal_actions.append('down')

    return rng.choice(optimal_actions)
