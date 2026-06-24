#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.pretrained import PreTrainedPolicy


def make_optimizer_and_scheduler(
    cfg: TrainPipelineConfig,
    policy: PreTrainedPolicy,
    effective_warmup_steps: int | None = None,
    effective_decay_steps: int | None = None,
) -> tuple[Optimizer, LRScheduler | None]:
    """Generates the optimizer and scheduler based on configs.

    Args:
        cfg (TrainPipelineConfig): The training config that contains optimizer and scheduler configs
        policy (PreTrainedPolicy): The policy config from which parameters and presets must be taken from.
        effective_warmup_steps: Override warmup steps (used when resuming to restore the original lambda).
        effective_decay_steps: Override decay steps (used when resuming to restore the original lambda).

    Returns:
        tuple[Optimizer, LRScheduler | None]: The couple (Optimizer, Scheduler). Scheduler can be `None`.
    """
    params = policy.get_optim_params() if cfg.use_policy_training_preset else policy.parameters()
    if cfg.optimizer is None:
        raise ValueError("Optimizer config is required but not provided in TrainPipelineConfig")
    optimizer = cfg.optimizer.build(params)
    if cfg.scheduler is not None:
        lr_scheduler = cfg.scheduler.build(
            optimizer,
            cfg.steps,
            effective_warmup_steps=effective_warmup_steps,
            effective_decay_steps=effective_decay_steps,
        )
    else:
        lr_scheduler = None
    return optimizer, lr_scheduler
