"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import dtorch


class MainNode:
    @staticmethod
    def set_address(address: str) -> None:
        return dtorch._dtorch_py_api.cluster._main_node_set_address(address)

    @staticmethod
    def wait_cluster_ready(num_nodes: int, timeout_second: float = 600) -> None:
        return dtorch._dtorch_py_api.cluster._main_node_wait_cluster_ready(num_nodes, timeout_second)

    @staticmethod
    def get_address() -> str:
        return dtorch._dtorch_py_api.cluster._main_node_get_address()

    @staticmethod
    def num_node_in_cluster() -> int:
        return dtorch._dtorch_py_api.cluster._main_node_num_node_in_cluster()
