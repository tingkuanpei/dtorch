"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

from typing import overload, Sequence, Union

import torch

import dtorch


class OperatorFormat:
    """Python wrapper around C++ OperatorFormat enum for nicer __str__ / class-level enum access."""

    def __init__(self, cpp_value: dtorch._dtorch_py_api.OperatorFormat) -> None:
        super(OperatorFormat, self).__init__()
        self._value = cpp_value

    @property
    def value(self) -> int:
        return self._value.value

    @property
    def name(self) -> str:
        return self._value.name

    def __str__(self) -> str:
        return "dtorch.OperatorFormat(" + self._value._to_string() + ")"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (OperatorFormat, dtorch._dtorch_py_api.OperatorFormat)):
            return False
        return self.value == other.value

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.value)


class PaddingType:
    """Python wrapper around C++ PaddingType enum for nicer __str__ / class-level enum access."""

    def __init__(self, cpp_value: dtorch._dtorch_py_api.PaddingType) -> None:
        super(PaddingType, self).__init__()
        self._value = cpp_value

    @property
    def value(self) -> int:
        return self._value.value

    @property
    def name(self) -> str:
        return self._value.name

    def __str__(self) -> str:
        return "dtorch.PaddingType(" + self._value._to_string() + ")"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (PaddingType, dtorch._dtorch_py_api.PaddingType)):
            return False
        return self.value == other.value

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.value)


class PoolingKind:
    """Python wrapper around C++ PoolingKind enum for nicer __str__ / class-level enum access."""

    def __init__(self, cpp_value: dtorch._dtorch_py_api.PoolingKind) -> None:
        super(PoolingKind, self).__init__()
        self._value = cpp_value

    @property
    def value(self) -> int:
        return self._value.value

    @property
    def name(self) -> str:
        return self._value.name

    def __str__(self) -> str:
        return "dtorch.PoolingKind(" + self._value._to_string() + ")"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (PoolingKind, dtorch._dtorch_py_api.PoolingKind)):
            return False
        return self.value == other.value

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.value)


# Set up class-level enum value attributes as Python wrapper instances
for _cls, _cpp_type in [
    (OperatorFormat, dtorch._dtorch_py_api.OperatorFormat),
    (PaddingType, dtorch._dtorch_py_api.PaddingType),
    (PoolingKind, dtorch._dtorch_py_api.PoolingKind),
]:
    for _name, _value in _cpp_type.__members__.items():
        setattr(_cls, _name, _cls(_value))
