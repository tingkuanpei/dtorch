"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

from typing import Optional

import dtorch


class MemoryStat(dtorch._dtorch_py_api.MemoryStat):
    def __init__(
        self,
        cpp_instance=None,
        *,
        allocated: int = 0,
        reserved: int = 0,
        max_memory_allocated: int = 0,
        max_memory_reserved: int = 0
    ) -> None:
        if cpp_instance is not None:
            super().__init__(cpp_instance)
            return
        super().__init__()
        self._set_allocated(allocated)
        self._set_reserved(reserved)
        self._set_max_memory_allocated(max_memory_allocated)
        self._set_max_memory_reserved(max_memory_reserved)

    @property
    def memory_allocated(self) -> int:
        return self._get_allocated()

    @memory_allocated.setter
    def memory_allocated(self, value: int):
        self._set_allocated(value)

    @property
    def memory_reserved(self) -> int:
        return self._get_reserved()

    @memory_reserved.setter
    def memory_reserved(self, value: int):
        self._set_reserved(value)

    @property
    def max_memory_allocated(self) -> int:
        return self._get_max_allocated()

    @max_memory_allocated.setter
    def max_memory_allocated(self, value: int):
        self._set_max_allocated(value)

    @property
    def max_memory_reserved(self) -> int:
        return self._get_max_reserved()

    @max_memory_reserved.setter
    def max_memory_reserved(self, value: int):
        self._set_max_reserved(value)

    def __str__(self) -> str:
        return self._to_string()

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: "MemoryStat") -> bool:
        return self._is_equal(other)


class MemoryStats(dtorch._dtorch_py_api.MemoryStats):
    def __init__(self, cpp_instance=None) -> None:
        if cpp_instance is not None:
            super().__init__(cpp_instance)
            return
        super().__init__()

    @property
    def size(self) -> int:
        return self._get_size()

    def find(self, device_id: int) -> Optional[MemoryStat]:
        return MemoryStat(self._find(device_id))

    def __str__(self) -> str:
        return self._to_string()

    def __repr__(self) -> str:
        return self.__str__()
