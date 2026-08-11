"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import os
import glob
import json
import re
import subprocess
import sys
from typing import Optional
import shutil
import time

import triton
from triton.backends.nvidia.driver import include_dirs, library_dirs

# Reference: https://github.com/triton-lang/triton/blob/v3.1.0/python/test/unit/tools/test_aot.py


class KernelBundle:
    def __init__(
        self,
        file_name: str,
        src_code: str,
        name: str = "",
        sig: str = "",
        grid: str = "",
        num_warps: Optional[int] = None,
    ):
        self.file_name = file_name
        self.src_code = src_code
        self.name = name
        self.sig = sig
        self.grid = grid
        self.num_warps = num_warps

    def __str__(self) -> str:
        return (
            f"\nKernelBundle {{\n    file_name: {self.file_name}\n    src_code: ***\n    name: {self.name}"
            f"\n    sig: {self.sig}\n    grid: {self.grid}\n    num_warps: {self.num_warps}\n}}"
        )

    def __repr__(self) -> str:
        return self.__str__()


class TritonKernelBundles:
    def __init__(self, root_path: str):
        triton_kernel_files = glob.glob(glob.os.path.join(root_path, "**", "*triton_kernel.cc"), recursive=True)

        triton_kernel_bundles = {}
        triton_kernel_srcs = {}
        for triton_kernel_file in triton_kernel_files:
            with open(triton_kernel_file, "r") as file:
                file_content = file.read()
                matches = re.findall(r'const std::string(.*?)\)";', file_content, re.DOTALL)
                for match in matches:
                    match_split = match.split('= R"(')
                    assert len(match_split) == 2

                    key = match_split[0].strip()
                    value = match_split[1].strip()
                    if key in triton_kernel_bundles or key in triton_kernel_srcs:
                        raise KeyError(f"Duplicate variable name: {key}")
                    if not key.endswith("TritonKernelBundle") and not key.endswith("TritonKernelSrc"):
                        continue

                    if key.endswith("TritonKernelBundle"):
                        try:
                            triton_kernel_bundles[key] = json.loads(value)
                        except json.JSONDecodeError as e:
                            raise KeyError(f"Parse json error: {e}, string: {value}")
                    else:
                        triton_kernel_srcs[key] = value

        for name, kernel_bundle in triton_kernel_bundles.items():
            assert "src_code" in kernel_bundle
            assert kernel_bundle["src_code"][0] == "$"
            kernel_src_code_name = kernel_bundle["src_code"][1:]
            assert kernel_src_code_name in triton_kernel_srcs
            kernel_bundle["src_code"] = triton_kernel_srcs[kernel_src_code_name]

        self.kernel_bundles = [KernelBundle(**value) for key, value in triton_kernel_bundles.items()]

    def write_kernel_to_file(self, root_dir):
        for bundle in self.kernel_bundles:
            kernel_path = os.path.join(root_dir, bundle.file_name)
            with open(kernel_path, "a") as file:
                file.write(bundle.src_code)


class TritonAotCompiler:
    def compile_aot_kernel(
        self,
        compile_dir,
        signature,
        kernel_name,
        out_name,
        out_path,
        num_warps,
        grid,
        kernel_path,
    ):
        compiler_path = os.path.join(triton.tools.__path__[0], "compile.py")

        subprocess.run(
            [
                sys.executable,
                compiler_path,
                "-n",
                kernel_name,
                "--signature",
                signature,
                "--out-name",
                out_name,
                "-o",
                out_path,
                "-w",
                str(num_warps),
                "-g",
                grid,
                kernel_path,
            ],
            check=True,
            cwd=compile_dir,
        )

    def link_aot_kernels(self, compile_dir):
        linker_path = os.path.join(triton.tools.__path__[0], "link.py")

        # link all desired configs
        h_files = glob.glob(os.path.join(compile_dir, "*.h"))
        h_files = [os.path.basename(file_name) for file_name in h_files]

        subprocess.run(
            [sys.executable, linker_path] + h_files + ["-o", "triton_aot_kernel"],
            check=True,
            cwd=compile_dir,
        )

    def gen_kernel_library(
        self,
        compile_dir,
        cpp_compile_command: str = None,
        library_name: str = "libtriton_aot_kernel.so",
    ):
        c_files = glob.glob(os.path.join(compile_dir, "*.c"))
        c_files = [os.path.basename(file_name) for file_name in c_files]

        if cpp_compile_command is None:
            cpp_compile_command = "gcc"
        cpp_compile_command = cpp_compile_command.replace("-Werror=return-type", "")
        cpp_compile_command = cpp_compile_command.replace("  ", " ")
        cpp_compile_command = cpp_compile_command.split(" ")
        assert len(cpp_compile_command) > 0

        subprocess.run(
            cpp_compile_command + c_files + ["-O3", "-I", include_dirs[0], "-c", "-fPIC"],
            check=True,
            cwd=compile_dir,
        )
        o_files = glob.glob(os.path.join(compile_dir, "*.o"))
        o_files = [os.path.basename(file_name) for file_name in o_files]

        command = [cpp_compile_command[0], *o_files, "-shared", "-o", library_name]
        for lib_dir in library_dirs():
            command.extend(["-L", lib_dir])
        subprocess.run(command, check=True, cwd=compile_dir)

    def recreate_folder(self, folder_path: str):
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

        os.makedirs(folder_path)

    def need_compile(self, root_path: str, output_path: str):
        # Check need to regenereate file by modified time
        need_to_regenerate = False
        current_time_seconds = time.time()

        if os.path.exists(output_path):
            output_path_modify_time_seconds = os.path.getmtime(output_path)
            if output_path_modify_time_seconds < os.path.getmtime(os.path.abspath(__file__)):
                need_to_regenerate = True
            elif current_time_seconds < output_path_modify_time_seconds:
                need_to_regenerate = True
            else:
                triton_kernel_files = glob.glob(
                    glob.os.path.join(root_path, "**", "*triton_kernel.cc"),
                    recursive=True,
                )
                for triton_kernel in triton_kernel_files:
                    triton_kernel_modify_time_seconds = os.path.getmtime(triton_kernel)
                    if triton_kernel_modify_time_seconds > output_path_modify_time_seconds:
                        need_to_regenerate = True
                        break
        else:
            need_to_regenerate = True

        return need_to_regenerate

    def compile(self, root_path: str, output_path: str, cpp_compile_command: str = None):
        if not self.need_compile(root_path, output_path):
            return

        self.recreate_folder(output_path)

        triton_kernel_bundles = TritonKernelBundles(root_path)
        triton_kernel_bundles.write_kernel_to_file(output_path)
        for bundle in triton_kernel_bundles.kernel_bundles:
            if not bundle.sig:
                continue
            self.compile_aot_kernel(
                compile_dir=output_path,
                signature=bundle.sig,
                kernel_name=bundle.name,
                out_name=bundle.name,
                out_path=bundle.name,
                num_warps=bundle.num_warps,
                grid=bundle.grid,
                kernel_path=bundle.file_name,
            )
        self.link_aot_kernels(output_path)
        self.gen_kernel_library(output_path, cpp_compile_command)


def main():
    if len(sys.argv) < 3:
        print("Need more arguments, eg: python triron_compile.py ./ ./build-linux/triton_compile_dir")
        sys.exit(1)

    project_root = sys.argv[1]
    project_binary_root = sys.argv[2]
    cpp_compile_command = sys.argv[3] if len(sys.argv) > 3 else None

    triton_compiler = TritonAotCompiler()
    triton_compiler.compile(project_root, project_binary_root, cpp_compile_command)

    # cmake list for include_dir and library_dirs
    include_dir_str = " ".join(include_dirs)
    library_dirs_str = [f"-L{dir}" for dir in library_dirs()]
    library_dirs_str.append("-lcuda")
    library_dirs_str = " ".join(library_dirs_str)
    print(f"{include_dir_str};{library_dirs_str}")


if __name__ == "__main__":
    sys.exit(main())
