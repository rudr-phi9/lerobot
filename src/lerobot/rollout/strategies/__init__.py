# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Rollout strategy ABC, factory, and shared inference helper."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import torch

from lerobot.policies.rtc import ActionInterpolator
from lerobot.policies.utils import make_robot_action
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame

if TYPE_CHECKING:
    from lerobot.rollout.configs import RolloutStrategyConfig
    from lerobot.rollout.context import RolloutContext
    from lerobot.rollout.inference import InferenceEngine


class RolloutStrategy(abc.ABC):
    """Abstract base for rollout execution strategies.

    Each concrete strategy implements a self-contained control loop with
    its own recording/interaction semantics.  Strategies are mutually
    exclusive — only one runs per session.
    """

    def __init__(self, config: RolloutStrategyConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def setup(self, ctx: RolloutContext) -> None:
        """Strategy-specific initialisation (keyboard listeners, buffers, etc.)."""

    @abc.abstractmethod
    def run(self, ctx: RolloutContext) -> None:
        """Main rollout loop.  Returns when shutdown is requested or duration expires."""

    @abc.abstractmethod
    def teardown(self, ctx: RolloutContext) -> None:
        """Cleanup: save dataset, stop threads, disconnect hardware."""


# ---------------------------------------------------------------------------
# Shared inference helper
# ---------------------------------------------------------------------------


def infer_action(
    engine: InferenceEngine,
    obs_processed: dict,
    obs_raw: dict,
    ctx: RolloutContext,
    interpolator: ActionInterpolator,
    ordered_keys: list[str],
    features: dict,
) -> dict | None:
    """Run one policy inference step and send the resulting action to the robot.

    Handles both sync and RTC backends.  Uses the interpolator for smooth
    control at higher-than-inference rates (works with any multiplier,
    including 1 where it acts as a pass-through).

    Parameters
    ----------
    engine:
        The inference engine (sync or RTC).
    obs_processed:
        Observation dict after ``robot_observation_processor``.
    obs_raw:
        Raw observation dict (needed by ``robot_action_processor``).
    ctx:
        Rollout context.
    interpolator:
        Action interpolator for Nx control rate.
    ordered_keys:
        Ordered action feature names (policy-to-robot mapping).
    features:
        Feature specification dict for ``build_dataset_frame`` /
        ``make_robot_action``.  Use ``dataset.features`` when recording,
        ``ctx.dataset_features`` otherwise.

    Returns
    -------
    Action dict sent to the robot, or ``None`` if no action was
    available (empty RTC queue, interpolator buffer not ready).
    """
    if engine.is_rtc:
        if interpolator.needs_new_action():
            action_tensor = engine.consume_rtc_action()
            if action_tensor is not None:
                interpolator.add(action_tensor.cpu())
    else:
        if interpolator.needs_new_action():
            obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
            action_tensor = engine.get_action_sync(obs_frame)
            action_dict = make_robot_action(action_tensor, features)
            action_t = torch.tensor([action_dict[k] for k in ordered_keys])
            interpolator.add(action_t)

    interp = interpolator.get()
    if interp is not None:
        action_dict = {k: interp[i].item() for i, k in enumerate(ordered_keys) if i < len(interp)}
        processed = ctx.robot_action_processor((action_dict, obs_raw))
        ctx.robot_wrapper.send_action(processed)
        return action_dict
    return None


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------


def create_strategy(config: RolloutStrategyConfig) -> RolloutStrategy:
    """Instantiate the appropriate strategy from a config object."""
    from lerobot.rollout.configs import (
        BaseStrategyConfig,
        DAggerStrategyConfig,
        HighlightStrategyConfig,
        SentryStrategyConfig,
    )

    if isinstance(config, BaseStrategyConfig):
        from .base import BaseStrategy

        return BaseStrategy(config)
    if isinstance(config, SentryStrategyConfig):
        from .sentry import SentryStrategy

        return SentryStrategy(config)
    if isinstance(config, HighlightStrategyConfig):
        from .highlight import HighlightStrategy

        return HighlightStrategy(config)
    if isinstance(config, DAggerStrategyConfig):
        from .dagger import DAggerStrategy

        return DAggerStrategy(config)

    raise ValueError(f"Unknown strategy config type: {type(config).__name__}")
