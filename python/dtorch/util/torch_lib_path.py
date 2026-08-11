import os
import platform
from pathlib import Path


def _setup_torch_lib_path():
    assert platform.system() == "Linux"

    try:
        import torch

        torch_dir = Path(torch.__file__).parent

        torch_lib_dirs = [
            torch_dir / "lib",  # 主要路径
            torch_dir / "lib64",
            torch_dir.parent / "torch/lib",
        ]

        lib_dir = None
        for dir_path in torch_lib_dirs:
            if dir_path.exists() and any(dir_path.glob("libtorch*.so")):
                lib_dir = str(dir_path)
                break
        if not lib_dir:
            raise FileNotFoundError("Cannot find PyTorch lib directory with libtorch.so")

        env_var = "LD_LIBRARY_PATH"
        current_path = os.environ.get(env_var, "")
        if lib_dir not in current_path:
            os.environ[env_var] = f"{lib_dir}:{current_path}"
    except Exception as e:
        raise RuntimeError(f"Failed to setup PyTorch lib path: {e}")


_setup_torch_lib_path()
