#!/usr/bin/env python

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

import numpy as np

from lerobot.envs.vlabench import ACTION_HIGH, ACTION_LOW, VLABenchEnv


def test_vlabench_action_space_uses_zero_to_one_gripper_channel():
    env = VLABenchEnv()

    np.testing.assert_array_equal(env.action_space.low, ACTION_LOW)
    np.testing.assert_array_equal(env.action_space.high, ACTION_HIGH)
    assert env.action_space.shape == (7,)
    assert env.action_space.low[-1] == 0.0
    assert env.action_space.high[-1] == 1.0
    np.testing.assert_array_equal(env.action_space.low[:6], np.full(6, -1.0, dtype=np.float32))
    np.testing.assert_array_equal(env.action_space.high[:6], np.full(6, 1.0, dtype=np.float32))


def test_vlabench_gripper_action_normalization_keeps_backward_compatibility():
    assert VLABenchEnv._normalize_gripper_action(-1.0) == 0.0
    assert VLABenchEnv._normalize_gripper_action(-0.5) == 0.25
    assert VLABenchEnv._normalize_gripper_action(0.0) == 0.0
    assert VLABenchEnv._normalize_gripper_action(0.5) == 0.5
    assert VLABenchEnv._normalize_gripper_action(1.0) == 1.0
    assert VLABenchEnv._normalize_gripper_action(1.5) == 1.0
