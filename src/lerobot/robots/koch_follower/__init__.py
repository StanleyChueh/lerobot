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

from .config_koch_follower import KochFollowerConfig, KochFollowerEndEffectorConfig
from .koch_follower import KochFollower
from .koch_follower_end_effector import KochFollowerEndEffector


# from .config_koch_follower import KochFollowerConfig
# from .koch_follower import KochFollower

# from .config_koch_follower import KochFollowerEndEffectorConfig
# import numpy as np

# class KochFollowerEndEffector(KochFollower):
#     """Same driver as KochFollower, but paired with URDF/EE-aware config."""
#     config_class = KochFollowerEndEffectorConfig
#     name = "koch_follower_end_effector"

# class KochFollowerEndEffector(KochFollower):
#     """Same driver as KochFollower, but paired with URDF/EE-aware config."""
#     # 你之前應該已有：
#     # config_class = KochFollowerEndEffectorConfig
#     # name = "koch_follower_end_effector"

#     @property
#     def end_effector_bounds(self) -> dict[str, np.ndarray]:
#         """
#         EEObservationWrapper 需要 {'min': np.ndarray(3,), 'max': np.ndarray(3,)}
#         這裡把 config 裡的 list 轉成 np.ndarray，dtype=float32。
#         """
#         cfg = self.config
#         return {
#             "min": np.asarray(cfg.end_effector_bounds["min"], dtype=np.float32),
#             "max": np.asarray(cfg.end_effector_bounds["max"], dtype=np.float32),
#         }