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
from typing import Any, Optional, Union
import inspect
from tqdm import tqdm
import asyncio


class DiffusionPipeline:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        model_path = pretrained_model_name_or_path
        if not os.path.isdir(model_path):
            raise ValueError(
                f"pretrained_model_name_or_path MUST be directory, "
                f"not support download model from internet: {model_path}"
            )

        model_index_path = os.path.join(model_path, "model_index.json")
        if not os.path.exists(model_index_path):
            raise ValueError(f"Can't find model_index.json in {model_path}")
        with open(model_index_path, "r") as file:
            model_index_data = json.load(file)

        from transformers import (
            CLIPTokenizer,
            T5TokenizerFast,
            CLIPTokenizer,
        )

        from dtorch.applications.huggingface.transformers import (
            CLIPTextModel,
            CLIPTextModelWithProjection,
            T5EncoderModel,
        )

        from dtorch.applications.huggingface.diffusers import (
            FlowMatchEulerDiscreteScheduler,
            AutoencoderKL,
            SD3Transformer2DModel,
            FluxTransformer2DModel,
        )

        class_maping = {
            "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
            "CLIPTokenizer": CLIPTokenizer,
            "CLIPTokenizer": CLIPTokenizer,
            "T5EncoderModel": T5EncoderModel,
            "CLIPTextModelWithProjection": CLIPTextModelWithProjection,
            "SD3Transformer2DModel": SD3Transformer2DModel,
            "CLIPTextModelWithProjection": CLIPTextModelWithProjection,
            "T5TokenizerFast": T5TokenizerFast,
            "AutoencoderKL": AutoencoderKL,
            "FluxTransformer2DModel": FluxTransformer2DModel,
            "CLIPTextModel": CLIPTextModel,
        }

        init_func_param_key = inspect.signature(cls.__init__).parameters.keys()
        init_func_param_key = model_index_data.keys() & init_func_param_key
        model_class_map = {}
        for key in init_func_param_key:
            model_class_name = model_index_data[key]
            model_init_path = os.path.join(model_path, key)
            model_class_map[key] = (class_maping[model_class_name[-1]], model_init_path)

        pipeline = cls.init_pipeline(kwargs, model_class_map)

        return pipeline

    @classmethod
    def init_pipeline(cls, kwargs, model_class_map):
        init_kwargs = {}
        for key, (class_name, model_path) in model_class_map.items():
            model = class_name.from_pretrained(model_path, **kwargs)
            init_kwargs[key] = model

        return cls(**init_kwargs, **kwargs)

    @property
    def _execution_device(self):
        for param in self.transformer.parameters():
            return param.device
        raise KeyError("Can't get self.transformer")

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def __call__(self, *args, **kwargs):
        generator = self.call_imp(*args, **kwargs)
        while True:
            try:
                next(generator)
            except StopIteration as e:
                return e.value

    async def async_call(self, *args, **kwargs):
        generator = self.call_imp(*args, **kwargs)
        while True:
            try:
                next(generator)
                await asyncio.sleep(0)
            except StopIteration as e:
                return e.value

    def call_imp(self, *input: Any) -> None:
        raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required "call_imp" function')
