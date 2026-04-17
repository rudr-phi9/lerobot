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

"""Policy deployment engine with pluggable rollout strategies."""

from .configs import (
    BaseStrategyConfig,
    DAggerKeyboardConfig,
    DAggerPedalConfig,
    DAggerStrategyConfig,
    DatasetRecordConfig,
    HighlightStrategyConfig,
    RolloutConfig,
    RolloutStrategyConfig,
    SentryStrategyConfig,
)
from .context import RolloutContext, build_rollout_context
from .inference import (
    InferenceEngine,
    InferenceEngineConfig,
    RTCInferenceConfig,
    RTCInferenceEngine,
    SyncInferenceConfig,
    SyncInferenceEngine,
    create_inference_engine,
)
from .ring_buffer import RolloutRingBuffer
from .robot_wrapper import ThreadSafeRobot
from .strategies import RolloutStrategy, create_strategy

__all__ = [
    "BaseStrategyConfig",
    "DAggerKeyboardConfig",
    "DAggerPedalConfig",
    "DAggerStrategyConfig",
    "HighlightStrategyConfig",
    "InferenceEngine",
    "InferenceEngineConfig",
    "RTCInferenceConfig",
    "RTCInferenceEngine",
    "RolloutConfig",
    "RolloutContext",
    "DatasetRecordConfig",
    "RolloutRingBuffer",
    "RolloutStrategy",
    "RolloutStrategyConfig",
    "SentryStrategyConfig",
    "SyncInferenceConfig",
    "SyncInferenceEngine",
    "ThreadSafeRobot",
    "build_rollout_context",
    "create_inference_engine",
    "create_strategy",
]
