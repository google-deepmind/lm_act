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

"""Agent that random chooses a legal action."""

import dataclasses
from typing import Any

import numpy as np

from lm_act.src import config as config_lib
from lm_act.src import interfaces


@dataclasses.dataclass(frozen=True, kw_only=True)
class RandomAgentConfig(config_lib.Agent):
  """Configuration for the random agent."""

  name: str = 'random'


class RandomAgent(interfaces.Agent):
  """Random agent."""

  def __init__(self, config: RandomAgentConfig) -> None:
    pass

  def step(
      self,
      observation: Any,
      environment: interfaces.Environment,
      rng: np.random.Generator,
  ) -> str:
    return environment.sample_legal_action(rng)
