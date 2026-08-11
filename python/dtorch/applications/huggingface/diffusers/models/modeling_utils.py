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

import dtorch
from safetensors.torch import load_file
from diffusers.utils import CONFIG_NAME


class ModelMixin(dtorch.nn.Module):
    config_name = CONFIG_NAME

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        torch_dtype = kwargs.pop("torch_dtype", None)

        model_path = pretrained_model_name_or_path
        with open(os.path.join(model_path, cls.config_name), "r", encoding="utf-8") as f:
            config = json.load(f)

        with dtorch.default_graph.dtype_guard(torch_dtype):
            model = cls(**config, **kwargs)

        state_dict = cls._load_state_dict(model_path)
        # TODO: remove strict=False
        model.load_state_dict(state_dict, strict=False)

        return model

    @classmethod
    def _load_state_dict(
        cls,
        model_path: Union[str, os.PathLike],
    ):
        all_files = []
        have_index_file = False
        index_file = ""
        for file in os.listdir(model_path):
            if os.path.isfile(os.path.join(model_path, file)):
                all_files.append(file)
                if file.endswith(".index.json"):
                    if have_index_file == True:
                        raise KeyError(f"Find too many .index.json file, {index_file} vs {file} ")
                    have_index_file = True
                    index_file = file

        state_dict_file_list = []
        if have_index_file:
            state_dict_map = {}
            with open(os.path.join(model_path, index_file), "r", encoding="utf-8") as f:
                state_dict_map = json.load(f)

            state_dict_file_list = [
                os.path.join(model_path, value) for key, value in state_dict_map["weight_map"].items()
            ]
            state_dict_file_list = list(set(state_dict_file_list))
        else:
            for file in all_files:
                if file.endswith(".safetensors") and not file.endswith(".fp16.safetensors"):
                    state_dict_file_list.append(os.path.join(model_path, file))
            if len(state_dict_file_list) != 1:
                raise KeyError(f"Number of state dict file not equal to 1: {state_dict_file_list}")

        result = {}
        for state_dict_file in state_dict_file_list:
            try:
                result.update(load_file(state_dict_file))
            except Exception as e:
                print(f"load model file error: {e}")
        return result
