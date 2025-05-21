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

"""Builds the prompts for the experiment."""

from lm_act.src import config as config_lib


def build_demonstration_prompt(
    demonstrations: str,
) -> str:
  """Returns the prompt for the demonstrations."""
  if not demonstrations:
    return (
        'You are an intelligent agent operating in a dynamic environment. Based'
        ' on the series of observations provided, you need to determine the'
        ' optimal action that maximizes the expected reward or achieves the'
        ' desired goal. Carefully consider all the given observations, infer'
        ' the current state of the environment, and select the most appropriate'
        ' action.\n\n'
    )

  return (
      'You are a powerful reinforcement learning agent. You can effectively'
      ' identify a policy exposed by demonstrations and reproduce it in a new'
      ' situation.\n\nHere are a number of'
      f' demonstrations:\n\n{demonstrations}\n'
  )


def build_trajectory_prompt(
    trajectory: str,
    legal_actions: list[str],
    config: config_lib.Experiment,
) -> str:
  """Returns the prompt for the current trajectory."""
  prompt = f'\nThis is the current trajectory:\n\n{trajectory}\n'

  if config.prompt.show_legal_actions:
    prompt += (
        '\nIn this situation, this is the list of all the actions that are'
        f' legal:\n\n{", ".join(legal_actions)}\n'
    )

  prompt += '\nGiven the '
  if 0 < config.num_demonstrations:
    prompt += 'demonstrations and the '
  prompt += 'current trajectory, you should infer the next logical action.'

  if config.prompt.show_legal_actions:
    prompt += '\nCheck that the chosen action is in the set of legal actions.'

  if config.prompt.use_chain_of_thought:
    prompt += (
        '\nThink step by step and very briefly explain your reasoning for'
        ' choosing this action.\nYou must answer with the reasoning followed by'
        ' the action in the following format:\nReasoning: ...\nAction: ...'
    )
  else:
    if config.prompt.show_legal_actions:
      prompt += (
          '\nYou must answer with one of the legal actions only, without any'
          ' other text.'
      )
    else:
      prompt += (
          '\nYou must answer with the action only, without any other text,'
          ' following exactly the same format as the previous actions.'
      )

  return prompt.strip()
