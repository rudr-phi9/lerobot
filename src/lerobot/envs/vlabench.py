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
"""VLABench environment wrapper for LeRobot.

VLABench is a large-scale benchmark for language-conditioned robotic manipulation
with long-horizon reasoning, built on MuJoCo/dm_control.

- Paper: https://arxiv.org/abs/2412.18194
- GitHub: https://github.com/OpenMOSS/VLABench
- Website: https://vlabench.github.io
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lerobot.types import RobotObservation

from .utils import _LazyAsyncVectorEnv

ACTION_DIM = 7  # pos(3) + euler(3) + gripper(1)
ACTION_LOW = -1.0
ACTION_HIGH = 1.0

# Default max episode steps per task type
DEFAULT_MAX_EPISODE_STEPS = 500

# VLABench task suites
PRIMITIVE_TASKS = [
    "select_fruit",
    "select_toy",
    "select_chemistry_tube",
    "add_condiment",
    "select_book",
    "select_painting",
    "select_drink",
    "insert_flower",
    "select_billiards",
    "select_ingredient",
    "select_mahjong",
    "select_poker",
    # Physical series
    "density_qa",
    "friction_qa",
    "magnetism_qa",
    "reflection_qa",
    "simple_cuestick_usage",
    "simple_seesaw_usage",
    "sound_speed_qa",
    "thermal_expansion_qa",
    "weight_qa",
]

COMPOSITE_TASKS = [
    "cluster_billiards",
    "cluster_book",
    "cluster_drink",
    "cluster_toy",
    "cook_dishes",
    "cool_drink",
    "find_unseen_object",
    "get_coffee",
    "hammer_nail",
    "heat_food",
    "make_juice",
    "play_mahjong",
    "play_math_game",
    "play_poker",
    "play_snooker",
    "rearrange_book",
    "rearrange_chemistry_tube",
    "set_dining_table",
    "set_study_table",
    "store_food",
    "take_chemistry_experiment",
    "use_seesaw_complex",
]

SUITE_TASKS: dict[str, list[str]] = {
    "primitive": PRIMITIVE_TASKS,
    "composite": COMPOSITE_TASKS,
}


class VLABenchEnv(gym.Env):
    """Gymnasium wrapper for VLABench environments.

    Wraps the dm_control-based VLABench simulator behind a standard gym.Env interface.
    Supports multiple cameras (front, second, wrist) and end-effector control.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        task: str = "select_fruit",
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        render_resolution: tuple[int, int] = (480, 480),
        robot: str = "franka",
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        action_mode: str = "eef",
        episode_index: int = 0,
        n_envs: int = 1,
    ):
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.render_resolution = render_resolution
        self.robot = robot
        self._max_episode_steps = max_episode_steps
        self.action_mode = action_mode
        self.episode_index = episode_index
        self.n_envs = n_envs

        # Deferred — created on first reset() inside worker subprocess to avoid
        # inheriting stale GPU/EGL contexts when AsyncVectorEnv spawns workers.
        self._env = None
        self._physics = None
        self.task_description = ""  # populated on first reset

        h, w = self.render_resolution

        if self.obs_type == "state":
            raise NotImplementedError(
                "The 'state' observation type is not supported in VLABenchEnv. "
                "Please use 'pixels' or 'pixels_agent_pos'."
            )
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "image": spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8),
                            "second_image": spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8),
                            "wrist_image": spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8),
                        }
                    ),
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "image": spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8),
                            "second_image": spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8),
                            "wrist_image": spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8),
                        }
                    ),
                    "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                }
            )
        else:
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(ACTION_DIM,), dtype=np.float32
        )

    def _ensure_env(self) -> None:
        """Create the underlying VLABench env on first use.

        Called inside the worker subprocess after fork(), so each worker gets
        its own clean rendering context rather than inheriting a stale one from
        the parent process (which causes crashes with AsyncVectorEnv).
        """
        if self._env is not None:
            return

        import VLABench.robots  # noqa: F401  # type: ignore[import-untyped]
        import VLABench.tasks  # noqa: F401  # type: ignore[import-untyped]
        from VLABench.envs import load_env  # type: ignore[import-untyped]

        h, w = self.render_resolution
        env = load_env(
            task=self.task,
            robot=self.robot,
            render_resolution=(h, w),
        )
        self._env = env
        self._physics = env.physics

        # Extract task description from the dm_control task
        task_obj = env.task
        if hasattr(task_obj, "task_description"):
            self.task_description = task_obj.task_description
        elif hasattr(task_obj, "language_instruction"):
            self.task_description = task_obj.language_instruction
        else:
            self.task_description = self.task

    def _get_obs(self) -> dict:
        """Get current observation from the environment."""
        assert self._env is not None

        obs = self._env.get_observation()
        h, w = self.render_resolution

        def _to_hwc3(arr: np.ndarray) -> np.ndarray:
            """Coerce any camera array to the declared (h, w, 3) uint8 shape."""
            import cv2

            a = np.asarray(arr)
            # Drop a leading singleton batch dim if present.
            while a.ndim > 3 and a.shape[0] == 1:
                a = a[0]
            if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[-1] not in (1, 3, 4):
                # CHW → HWC
                a = np.transpose(a, (1, 2, 0))
            if a.ndim == 2:
                a = np.stack([a] * 3, axis=-1)
            if a.ndim != 3:
                return np.zeros((h, w, 3), dtype=np.uint8)
            # Force 3 channels.
            if a.shape[-1] == 1:
                a = np.repeat(a, 3, axis=-1)
            elif a.shape[-1] == 4:
                a = a[..., :3]
            elif a.shape[-1] != 3:
                return np.zeros((h, w, 3), dtype=np.uint8)
            if a.shape[:2] != (h, w):
                a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
            return a.astype(np.uint8)

        # Extract camera images — VLABench returns (n_cameras, C, H, W) or individual arrays
        raw_frames: list[np.ndarray] = []
        if "rgb" in obs:
            rgb = obs["rgb"]
            if isinstance(rgb, np.ndarray):
                if rgb.ndim == 4:
                    raw_frames = [rgb[i] for i in range(rgb.shape[0])]
                elif rgb.ndim == 3:
                    raw_frames = [rgb]

        image_keys = ["image", "second_image", "wrist_image"]
        images: dict[str, np.ndarray] = {}
        for i, key in enumerate(image_keys):
            if i < len(raw_frames):
                images[key] = _to_hwc3(raw_frames[i])
            else:
                images[key] = np.zeros((h, w, 3), dtype=np.uint8)

        # Extract end-effector state — coerce to exactly (7,) so vector env concat
        # doesn't fail with shape-mismatch on buffer np.stack.
        ee_state = obs.get("ee_state", np.zeros(7, dtype=np.float64))
        ee_state = np.asarray(ee_state, dtype=np.float64).ravel()
        if ee_state.shape[0] != 7:
            fixed = np.zeros(7, dtype=np.float64)
            fixed[: min(7, ee_state.shape[0])] = ee_state[:7]
            ee_state = fixed

        if self.obs_type == "pixels":
            return {"pixels": images}
        elif self.obs_type == "pixels_agent_pos":
            return {
                "pixels": images,
                "agent_pos": ee_state.astype(np.float64),
            }
        else:
            raise ValueError(f"Unknown obs_type: {self.obs_type}")

    def reset(self, seed=None, **kwargs) -> tuple[RobotObservation, dict[str, Any]]:
        self._ensure_env()
        assert self._env is not None
        super().reset(seed=seed)

        self._env.reset()

        observation = self._get_obs()
        info = {"is_success": False}
        return observation, info

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        self._ensure_env()
        assert self._env is not None

        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )

        # VLABench's dm_control task does `data.ctrl[:] = action` without adapting
        # shapes. Franka in VLABench has 9 actuators (7 arm joints + 2 gripper
        # fingers), but our policy emits a 7D action. Pad with zeros / repeat the
        # last value so the broadcast succeeds.
        assert self._physics is not None
        ctrl_dim = int(self._physics.data.ctrl.shape[0])
        if action.shape[0] != ctrl_dim:
            padded = np.zeros(ctrl_dim, dtype=action.dtype)
            padded[: min(action.shape[0], ctrl_dim)] = action[:ctrl_dim]
            if action.shape[0] < ctrl_dim:
                # Repeat the last entry (typically the gripper command) for the
                # trailing extra actuators.
                padded[action.shape[0] :] = action[-1]
            action = padded

        if self.action_mode not in ("eef", "joint", "delta_eef"):
            raise ValueError(f"Unknown action_mode: {self.action_mode}")
        timestep = self._env.step(action)

        # Extract reward from dm_control timestep
        reward = float(timestep.reward) if timestep.reward is not None else 0.0

        # Check success via the task's termination condition
        is_success = False
        if hasattr(self._env, "task") and hasattr(self._env.task, "should_terminate_episode"):
            is_success = bool(self._env.task.should_terminate_episode(self._physics))

        terminated = is_success
        truncated = False
        info = {
            "task": self.task,
            "is_success": is_success,
        }

        observation = self._get_obs()

        if terminated:
            self.reset()

        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        self._ensure_env()
        obs = self._get_obs()
        return obs["pixels"]["image"]

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
            self._physics = None


# ---- Factory helpers ---------------------------------------------------------


def _make_env_fns(
    *,
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any],
) -> list[Callable[[], VLABenchEnv]]:
    """Build n_envs factory callables for a single task."""

    def _make_env(episode_index: int, **kwargs) -> VLABenchEnv:
        return VLABenchEnv(
            task=task,
            episode_index=episode_index,
            n_envs=n_envs,
            **kwargs,
        )

    fns: list[Callable[[], VLABenchEnv]] = []
    for episode_index in range(n_envs):
        fns.append(partial(_make_env, episode_index, **gym_kwargs))
    return fns


# ---- Main API ----------------------------------------------------------------


def create_vlabench_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> dict[str, dict[int, Any]]:
    """
    Create vectorized VLABench environments with a consistent return shape.

    Returns:
        dict[suite_name][task_id] -> vec_env (env_cls([...]) with exactly n_envs factories)

    Notes:
        - n_envs is the number of rollouts *per task* (episode_index = 0..n_envs-1).
        - `task` can be a suite name ("primitive", "composite"), a comma-separated list of
          suite names, or individual task names (e.g. "select_fruit,heat_food").
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    task_groups = [t.strip() for t in task.split(",") if t.strip()]
    if not task_groups:
        raise ValueError("`task` must contain at least one VLABench task or suite name.")

    print(f"Creating VLABench envs | task_groups={task_groups} | n_envs(per task)={n_envs}")

    is_async = env_cls is gym.vector.AsyncVectorEnv
    cached_obs_space = None
    cached_act_space = None
    out: dict[str, dict[int, Any]] = defaultdict(dict)

    for group in task_groups:
        # Check if it's a suite name, otherwise treat as individual task
        tasks = SUITE_TASKS.get(group, [group])

        for tid, task_name in enumerate(tasks):
            print(f"Building vec env | group={group} | task_id={tid} | task={task_name}")

            fns = _make_env_fns(
                task=task_name,
                n_envs=n_envs,
                gym_kwargs=gym_kwargs,
            )

            if is_async:
                lazy = _LazyAsyncVectorEnv(fns, cached_obs_space, cached_act_space)
                if cached_obs_space is None:
                    cached_obs_space = lazy.observation_space
                    cached_act_space = lazy.action_space
                out[group][tid] = lazy
            else:
                out[group][tid] = env_cls(fns)

    return {group: dict(task_map) for group, task_map in out.items()}
