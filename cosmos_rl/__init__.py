# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Some CUDA-12.8 policy environments intentionally pin PyTorch below the
# version required by a separately installed optional torchao package.
# Transformers and diffusers otherwise discover that package eagerly while
# importing ordinary (non-quantized) VLM code and fail before training starts.
# Hide only this incompatible optional dependency; compatible installations
# and actual torchao workloads are unchanged.
import importlib.metadata
import importlib.util

from packaging.version import Version


def _mask_incompatible_optional_torchao() -> None:
    try:
        torch_version = Version(importlib.metadata.version("torch").split("+")[0])
        torchao_version = Version(importlib.metadata.version("torchao").split("+")[0])
    except importlib.metadata.PackageNotFoundError:
        return
    if torch_version >= Version("2.11") or torchao_version < Version("0.14"):
        return
    original_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "torchao" or name.startswith("torchao."):
            return None
        return original_find_spec(name, *args, **kwargs)

    importlib.util.find_spec = find_spec


_mask_incompatible_optional_torchao()

from . import policy
from . import comm
from . import dispatcher
from . import rollout
from . import launcher
from . import utils
from . import colocated
from . import tools
from . import simulators
from . import reward
from . import reference

__all__ = [
    "policy",
    "comm",
    "dispatcher",
    "rollout",
    "launcher",
    "utils",
    "colocated",
    "tools",
    "simulators",
    "reward",
    "reference",
]
