"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import re
from typing import cast, Optional, Union, Sequence, Tuple, Any, Dict, Set
import math

import torch

import dtorch


class DeviceMesh(dtorch._dtorch_py_api.DeviceMesh):
    def __init__(
        self,
        arg0: Union[str, torch.device],
        mesh: torch.Tensor = None,
        mesh_dim_names: Sequence[str] = None,
    ) -> None:
        if mesh is None and mesh_dim_names is None:
            if isinstance(arg0, dtorch._dtorch_py_api.DeviceMesh):
                super().__init__(arg0)
            else:
                torch_device = torch.device(arg0)
                super().__init__(torch_device)
        else:
            if isinstance(mesh, torch.Tensor):
                mesh = mesh.cpu().detach().to(dtype=torch.int64)
            else:
                mesh = torch.tensor(mesh, device="cpu", dtype=torch.int64)

            super().__init__(arg0, mesh, mesh_dim_names)

    @property
    def is_distributed(self) -> bool:
        return self._is_distributed()

    @property
    def device_type(self) -> torch.device:
        device_str = self._get_device_str()
        return torch.device("cuda" if device_str == "gpu" else device_str)

    @property
    def mesh(self) -> torch.Tensor:
        return self._get_mesh()

    @property
    def dim_names(self) -> Sequence[str]:
        return self._get_mesh_dimension_names()

    def has_dim_name(self, dim_name: str) -> bool:
        return self._has_dimension_name(dim_name)

    def check_all_dim_names_in_set(self, dim_name_set: Set[str]) -> bool:
        if self.is_distributed and len(self.dim_names) == 0:
            raise ValueError(
                "DeviceMesh is distributed but dim_names is empty, " "please set dim_names when initializing DeviceMesh"
            )
        assert all(
            [name in set(dim_name_set) for name in self.dim_names]
        ), f"DeviceMesh's dim_names({self.dim_names}) not in set({dim_name_set})"

    def dim_name_index(self, dim_name: str, default_value: Optional[int] = None) -> int:
        if not self.has_dim_name(dim_name):
            return default_value
        else:
            return self._get_dimension_name_index(dim_name)

    @property
    def ndim(self) -> int:
        return len(self._get_mesh_shape())

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._get_mesh_shape())

    # def size(self, mesh_dim: Optional[int] = None) -> int:
    #     return self.mesh.numel() if mesh_dim is None else self.mesh.size(mesh_dim)

    def __repr__(self) -> str:
        return (
            f"DeviceMesh('{self.device_type}', dim_name: {self.dim_names}, shape: {self.shape},"
            f" data: {self.mesh.tolist()})"
        )

    def to_device(self) -> torch.device:
        assert not self.is_distributed, "Only local tensor support to_deivce()"
        return self.first_device()

    def first_device(self) -> torch.device:
        device_str = self._get_device_str()
        device_str = "cuda" if device_str == "gpu" else device_str

        device_id = self._get_mesh_data()
        assert len(device_id) >= 1
        if device_str == "cpu" and device_id[0] == 0:
            device_id = None
        else:
            device_id = device_id[0]

        return torch.device(device_str, device_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeviceMesh):
            return False
        if id(self) == id(other):
            return True
        else:
            return self._is_same(other)

    def unbind(self, dims: Union[Sequence[str], Sequence[int], str, int]) -> Sequence["DeviceMesh"]:
        if isinstance(dims, str) or not isinstance(dims, Sequence):
            dims = [dims]

        int_dims = []
        for dim in dims:
            if isinstance(dim, str):
                int_dim = self.dim_name_index(dim)
                if int_dim is not None:
                    int_dims.append(int_dim)
            else:
                assert isinstance(dim, int)
                int_dims.append(dim)

        if len(int_dims) == 0:
            return [self]
        else:
            return [DeviceMesh(it) for it in self._unbind(int_dims)]


class Placement(dtorch._dtorch_py_api.Placement):
    def __init__(self, placement: str) -> None:
        super().__init__(placement)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Placement):
            return False
        if id(self) == id(other):
            return True
        return self._equal(other)

    def __repr__(self) -> str:
        return self._to_string()

    def is_replicate(self) -> bool:
        return self._is_replicate()

    def is_partial(self) -> bool:
        return self._is_partial()

    def is_shard(self) -> bool:
        return self._is_shard()

    def get_shard_index(self) -> int:
        return self._get_shard_index()

    def has_sub_split_coordinates(self) -> bool:
        return self._has_sub_split_coordinates()

    def get_sub_split_coordinates(self) -> int:
        return self._get_sub_split_coordinates()


def Shard(dim: int) -> Placement:
    return Placement(f"S{dim}")


def Replicate() -> Placement:
    return Placement(f"R")


def Partial() -> Placement:
    return Placement(f"P")


def init_device_mesh(
    device_type: str,
    mesh_shape: Union[int, Tuple[int, ...]],
    mesh_dim_names: Sequence[str] = None,
) -> DeviceMesh:
    if isinstance(mesh_shape, int):
        mesh_shape = (mesh_shape,)

    mesh_shape = [it for it in mesh_shape if it is not None]
    if len(mesh_shape) == 0:
        mesh_shape = [1]
    assert len(mesh_shape) > 0 and math.prod(mesh_shape) > 0

    with torch.device("cpu"):
        mesh = torch.arange(math.prod(mesh_shape), dtype=torch.int).view(mesh_shape)
    return DeviceMesh(device_type, mesh, mesh_dim_names=mesh_dim_names)


def get_default_device_mesh(
    device: Optional[Union[torch.device, str]] = None,
    device_mesh: Optional[DeviceMesh] = None,
    graph: Optional["Graph"] = None,
) -> DeviceMesh:
    assert not (device is not None and device_mesh is not None), "Can't set device and device_mesh at same time"

    if device is not None:
        return DeviceMesh(device)
    elif device_mesh is not None:
        if isinstance(device_mesh, DeviceMesh):
            return device_mesh
        else:
            return DeviceMesh(device_mesh)
    else:
        graph = graph if graph is not None else dtorch.Graph.default_graph()
        return graph.default_device_mesh


def get_placements_from_dict(
    device_mesh: DeviceMesh,
    placements_dict: Dict[Union[str, int], Placement],
    original_placements: Sequence[Placement] = None,
    default_placement_mode: str = "raise_error",
) -> Sequence[Placement]:
    # 0. Check arguments
    assert default_placement_mode in ("raise_error", "replicate", "keep")

    if device_mesh.ndim == 0:
        return []

    if default_placement_mode == "keep":
        assert original_placements is not None
        assert (
            len(original_placements) == device_mesh.ndim
        ), f"{len(original_placements)=} not equal to {device_mesh.ndim=}"

    if len(placements_dict) == 0:
        if default_placement_mode == "replicate":
            return [Replicate()] * device_mesh.ndim
        elif default_placement_mode == "keep":
            assert original_placements is not None
            assert len(original_placements) == device_mesh.ndim
            return original_placements
        else:
            raise KeyError(
                f"Length of placements_dict is zero and default_placement_mode is not replicate"
                f" or keep: {default_placement_mode}"
            )

    all_key_string = all(isinstance(key, str) for key in placements_dict.keys())
    all_key_int = all(isinstance(key, int) for key in placements_dict.keys())
    assert all_key_string or all_key_int, "key of placements_dict MUST all string or all int"

    if all_key_string:
        dim_names = device_mesh.dim_names
        if len(dim_names) == 0:
            raise ValueError("Key of placements_dict is dim_name but device_mesh not have dim_name")
        assert len(dim_names) == device_mesh.ndim, "device_mesh dim_name size not equal to device_mesh.ndim"

    # 1. Get result
    result = []

    for i in range(device_mesh.ndim):
        if all_key_string:
            dim_names = device_mesh.dim_names
            assert len(dim_names) == device_mesh.ndim, "device_mesh dim_name size not equal to device_mesh.ndim"
            key = dim_names[i]
        else:
            assert all_key_int
            key = i

        if key in placements_dict:
            result.append(placements_dict[key])
        elif default_placement_mode == "replicate":
            result.append(Replicate())
        elif default_placement_mode == "keep":
            result.append(original_placements[i])
        else:
            if all_key_string:
                raise KeyError(
                    f"Dimension name key: {key} not in placements_dict. "
                    f"Invalid placements_dict: {placements_dict}, "
                    f"device_mesh.dim_names: {device_mesh.dim_names}"
                )
            else:
                raise KeyError(
                    f"Index key: {key} not in placements_dict. "
                    f"Invalid placements_dict: {placements_dict}, "
                    f"device_mesh.ndim: {device_mesh.ndim}"
                )

    return result
