"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import dtorch


class GlobalOption:
    """Read-only Python view of the C++ ``GlobalOption`` singleton.

    Values are read once from the ``DTORCH_*`` environment variables when the C++ singleton is first
    constructed, then cached. Set the relevant environment variable before ``import dtorch`` / first
    use to take effect.
    """

    @staticmethod
    def get_comm_timeout_second() -> int:
        return dtorch._dtorch_py_api._global_option_get_comm_timeout_second()

    @staticmethod
    def get_grpc_timeout_second() -> int:
        return dtorch._dtorch_py_api._global_option_get_grpc_timeout_second()

    @staticmethod
    def get_zmq_timeout_second() -> int:
        return dtorch._dtorch_py_api._global_option_get_zmq_timeout_second()

    @staticmethod
    def get_dtensor_in_same_device() -> bool:
        return dtorch._dtorch_py_api._global_option_get_dtensor_in_same_device()

    @staticmethod
    def get_per_device_per_process() -> bool:
        return dtorch._dtorch_py_api._global_option_get_per_device_per_process()

    @staticmethod
    def get_num_gpu_when_enable_dtensor_in_same_device() -> int:
        return dtorch._dtorch_py_api._global_option_get_num_gpu_when_enable_dtensor_in_same_device()

    @staticmethod
    def get_validate_kernel_input_output() -> bool:
        return dtorch._dtorch_py_api._global_option_get_validate_kernel_input_output()
