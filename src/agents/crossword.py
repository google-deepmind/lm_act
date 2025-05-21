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

"""Oracle agent for the crossword environment."""

from collections.abc import Mapping
import dataclasses
from typing import Any

import numpy as np

from lm_act.src import config as config_lib
from lm_act.src import interfaces


@dataclasses.dataclass(frozen=True, kw_only=True)
class OracleAgentConfig(config_lib.Agent):
  """Configuration for the oracle agent."""

  name: str = 'crossword_oracle'


class OracleAgent(interfaces.Agent):
  """Interface for agents."""

  def __init__(self, config: OracleAgentConfig) -> None:
    pass

  def step(
      self,
      observation: Mapping[str, Any],
      environment: interfaces.Environment,
      rng: np.random.Generator,
  ) -> str:
    """Returns one of the missing words."""
    solutions = observation['puzzle'].split('Solution:\n')[1].split('\n')
    solutions = [sol.split(' (')[0] for sol in solutions]
    return rng.choice(solutions)
