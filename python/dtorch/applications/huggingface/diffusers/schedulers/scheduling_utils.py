# Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
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

"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import json
import os
from typing import Optional, Union


SCHEDULER_CONFIG_NAME = "scheduler_config.json"


class SchedulerMixin:
    config_name = SCHEDULER_CONFIG_NAME

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
    ):
        model_path = pretrained_model_name_or_path
        with open(os.path.join(model_path, "scheduler_config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
        scheduler = cls(**config)
        return scheduler
