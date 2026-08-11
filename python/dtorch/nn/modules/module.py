"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import itertools
from typing import (
    Dict,
    Optional,
    Union,
    Any,
    Callable,
    Sequence,
)

import torch

import dtorch
from dtorch import Tensor, DeviceMesh
from dtorch.nn.parameter import Parameter, Buffer


class DTorchModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # torch.nn.Module.__init__ sets training=True; DTorch defaults to training=False.
        # Must keep: need to override training default after super().__init__().
        super().__setattr__("training", False)

    # Must keep: provides a default forward() that raises a descriptive NotImplementedError,
    # which is then overridden by subclasses. This makes DTorchModule.forward always resolve
    # to the subclass's actual forward implementation.
    forward: Callable[..., Any] = torch.nn.Module.forward

    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:
        # Must keep: validates isinstance(tensor, dtorch.Tensor) instead of torch.Tensor.
        if "_buffers" not in self.__dict__:
            raise AttributeError("cannot assign buffer before DTorchModule.__init__() call")
        elif not isinstance(name, str):
            raise TypeError(f"buffer name should be a string. Got {torch.typename(name)}")
        elif "." in name:
            raise KeyError('buffer name can\'t contain "."')
        elif name == "":
            raise KeyError('buffer name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError(f"attribute '{name}' already exists")
        elif tensor is not None and not isinstance(tensor, dtorch.Tensor):
            raise TypeError(
                f"cannot assign '{torch.typename(tensor)}' object to buffer '{name}' "
                "(dtorch Tensor or None required)"
            )
        else:
            self._buffers[name] = tensor
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        # Must keep: validates isinstance(param, dtorch.nn.Parameter) instead of torch.nn.Parameter.
        if "_parameters" not in self.__dict__:
            raise AttributeError("cannot assign parameter before DTorchModule.__init__() call")

        elif not isinstance(name, str):
            raise TypeError(f"parameter name should be a string. Got {torch.typename(name)}")
        elif "." in name:
            raise KeyError('parameter name can\'t contain "."')
        elif name == "":
            raise KeyError('parameter name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"attribute '{name}' already exists")

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(
                f"cannot assign '{torch.typename(param)}' object to parameter '{name}' "
                "(dtorch.nn.Parameter or None required)"
            )
        else:
            self._parameters[name] = param

    def get_parameter(self, target: str) -> "Parameter":
        # Must keep: validates isinstance(param, dtorch.nn.Parameter) instead of torch.nn.Parameter.
        # dtorch.nn.Parameter does NOT inherit torch.nn.Parameter, so torch's version would ALWAYS fail.
        module_path, _, param_name = target.rpartition(".")

        mod: dtorch.nn.DTorchModule = self.get_submodule(module_path)

        if not hasattr(mod, param_name):
            raise AttributeError(mod._get_name() + " has no attribute `" + param_name + "`")

        param: dtorch.nn.Parameter = getattr(mod, param_name)

        if not isinstance(param, dtorch.nn.Parameter):
            raise AttributeError("`" + param_name + "` is not an " "nn.Parameter")

        return param

    def _apply(self, fn, recurse=True):
        # Must keep: wraps the result in dtorch.nn.Parameter(param_applied), while torch's _apply
        # has complex set_data / shallow-copy logic that is incompatible with DTorch tensors.
        if recurse:
            for module in self.children():
                module._apply(fn)

        for key, param in self._parameters.items():
            if param is None:
                continue
            param_applied = fn(param)
            out_param = Parameter(param_applied)
            self._parameters[key] = out_param

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        return self

    def to(self, *args, **kwargs):
        # Must keep: uses DTorch's Tensor.parser_tensor_to_function_arg() to parse device_mesh/dtype
        # arguments, which is a different API from torch.nn.Module.to(device=..., dtype=...).
        device_mesh, dtype = Tensor.parser_tensor_to_function_arg(args, kwargs)
        if device_mesh == None and dtype == None:
            return

        def convert_func(tensor):
            if tensor == None:
                return
            return tensor.to(
                device_mesh=device_mesh,
                dtype=dtype if tensor.is_floating_point() else None,
            )

        return self._apply(convert_func)

    def redistribute_input(self, *args, **kwargs):
        r"""Redistribute input tensor.

        Can be overridden by subclasses if need.
        """
        # Must keep: DTorch-specific hook for distributed tensor redistribution before forward().
        # torch.nn.Module has no equivalent hook.
        return args, kwargs

    def redistribute_output(self, output):
        r"""Redistribute output tensor.

        Can be overridden by subclasses if need.
        """
        # Must keep: DTorch-specific hook for distributed tensor redistribution after forward().
        # torch.nn.Module has no equivalent hook.
        return output

    def __call__(self, *args, **kwargs):
        # Must keep: injects redistribute_input() before forward() and redistribute_output() after,
        # enabling automatic distributed tensor redistribution. torch.nn.Module.__call__ only
        # runs forward hooks and forward(), without these redistribution hooks.

        if self.redistribute_input is not None:
            tmp_out = self.redistribute_input(*args, **kwargs)
            assert (
                len(tmp_out) == 2 and isinstance(tmp_out[0], Sequence) and isinstance(tmp_out[1], Dict)
            ), f"Out of redistribute_input() MUST can unpack as *args, **kwargs, but get: {tmp_out}"
            args, kwargs = tmp_out

        output = self.forward(*args, **kwargs)

        if self.redistribute_output is not None:
            output = self.redistribute_output(output)

        return output

    def __setattr__(self, name: str, value: Union[Tensor, "DTorchModule"]) -> None:
        # Must keep: handles DTorch-specific types — dtorch.nn.Parameter, DTorchModule, and
        # dtorch.nn.Buffer. torch.nn.Module.__setattr__ only recognizes torch.nn.Parameter and
        # torch.nn.Module types, which are different type hierarchies.
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get("_parameters")
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError("cannot assign parameters before DTorchModule.__init__() call")
            remove_from(
                self.__dict__,
                self._buffers,
                self._modules,
                self._non_persistent_buffers_set,
            )
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(
                    f"cannot assign '{torch.typename(value)}' as parameter '{name}' "
                    "(torch.nn.Parameter or None expected)"
                )
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get("_modules")
            if isinstance(value, DTorchModule):
                if modules is None:
                    raise AttributeError("cannot assign module before DTorchModule.__init__() call")
                remove_from(
                    self.__dict__,
                    self._parameters,
                    self._buffers,
                    self._non_persistent_buffers_set,
                )
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError(
                        f"cannot assign '{torch.typename(value)}' as child module '{name}' "
                        "(torch.nn.DTorchModule or None expected)"
                    )
                for hook in _global_module_registration_hooks.values():
                    output = hook(self, name, value)
                    if output is not None:
                        value = output
                modules[name] = value
            else:
                buffers = self.__dict__.get("_buffers")
                if isinstance(value, Buffer) or buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, torch.Tensor):
                        raise TypeError(
                            f"cannot assign '{torch.typename(value)}' as buffer '{name}' "
                            "(torch.nn.Buffer, torch.Tensor or None expected)"
                        )
                    if isinstance(value, Buffer):
                        persistent = value.persistent
                    else:
                        persistent = name not in self._non_persistent_buffers_set
                    self.register_buffer(name, value, persistent)
                else:
                    # Use object.__setattr__ to avoid torch.nn.Module.__setattr__
                    # which would accept non-DTorch torch.nn.Module/torch.nn.Parameter.
                    object.__setattr__(self, name, value)

    def _load_from_state_dict(
        # Must keep: handles torch.Tensor → dtorch.Tensor conversion during parameter loading.
        # torch.nn.Module._load_from_state_dict works with torch.Tensor only.
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # TODO: support unexpected_keys
        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}
        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                assert isinstance(param, Tensor)
                if isinstance(input_param, Tensor):
                    param.copy_(input_param)
                else:
                    assert isinstance(input_param, torch.Tensor)
                    param.copy_(dtorch.Tensor(input_param))
            elif strict:
                missing_keys.append(key)

    def first_param_device_mesh(self) -> Optional[DeviceMesh]:
        # Must keep: DTorch-specific utility that returns the DeviceMesh of the first parameter.
        # torch.nn.Module has no concept of DeviceMesh.
        for param in self.parameters():
            return param.device_mesh
        return None


# To avoid confusion with torch.Module
Module = DTorchModule
