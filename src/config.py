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

"""Configuration for the experiment."""

import dataclasses


@dataclasses.dataclass(frozen=True, kw_only=True)
class Agent:
  """Configuration for the agent."""

  name: str
  action_type: str = 'txt'


@dataclasses.dataclass(frozen=True, kw_only=True)
class Environment:
  """Configuration for the environment."""

  name: str
  observation_type: str
  action_type: str = 'txt'


@dataclasses.dataclass(frozen=True, kw_only=True)
class Prompt:
  """Configuration for the prompt."""

  show_legal_actions: bool
  use_chain_of_thought: bool
  include_past_actions: bool = True


@dataclasses.dataclass(frozen=True, kw_only=True)
class Experiment:
  """Configuration for the experiment."""

  num_demonstrations: int = 0
  num_evaluation_steps: int = 100
  num_evaluation_episodes: int = 100
  replay_episode: bool = False

  agent: Agent
  environment: Environment
  prompt: Prompt

  def __post_init__(self):
    if self.agent.action_type != self.environment.action_type:
      raise ValueError('The agent and environment action types must match.')

    if self.replay_episode and self.num_demonstrations != 1:
      raise ValueError('Replaying an episode requires exactly 1 demonstration.')
