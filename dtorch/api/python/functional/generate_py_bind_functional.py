"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import os
import glob
import re
import sys
import time
import shutil

file_begin = """/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <nanobind/stl/vector.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "dtorch/api/python/nanobind_register.h"
#include "dtorch/api/python/nanobind_type_casters.h"
#include "dtorch/api/python/functional/py_function_wrapper_helpers.h"
#include "dtorch/api/cpp/functional/activation.h"
#include "dtorch/api/cpp/functional/math.h"
#include "dtorch/api/cpp/functional/neural_network.h"
#include "dtorch/api/cpp/functional/tensor_functional.h"
#include "dtorch/api/cpp/functional/system_functional.h"
#include "dtorch/api/cpp/functional/functional_option.h"
#include "dtorch/api/cpp/functional/fused_compile_op.h"
#include "dtorch/api/python/py_bind_graph.h"

namespace dtorch {
namespace api {
namespace py {

class PyBindFunctionalGenerated {
public:
    static std::string ModuleName() { return "nn.functional"; }

    static void RegisterFunc(nb::module_& m) {

        using api::cpp::DataKind;
        using api::cpp::Generator;
        using api::cpp::Graph;
        using api::cpp::Index;
        using api::cpp::MemoryStats;
        using api::cpp::OperatorFormat;
        using api::cpp::PaddingType;
        using api::cpp::PoolingKind;
        using api::cpp::Scalar;
        using api::cpp::Shape;
        using api::cpp::Tensor;
        using api::cpp::IntOrIntArray;
        using api::cpp::PlacementSeq;
        using api::cpp::DeviceMesh;
        using api::cpp::Device;
        using api::cpp::TensorFuture;
        using api::cpp::VoidFutureCollect;
        using api::cpp::functional::SdpaOption;
"""

file_end = """
    }
};

}  // namespace py
}  // namespace api
}  // namespace dtorch
"""


def to_lower_case_name(name):
    assert len(name) > 0
    lower_case_name = re.sub(r"([A-Z])", lambda x: "_" + x.group(1).lower(), name)
    lower_case_name = lower_case_name.lstrip("_")
    if name[0] == "_":
        lower_case_name = "_" + lower_case_name
    return lower_case_name


class FuncParam:
    def __init__(self, cpp_param_string: str):
        self.is_optional = "std::optional" in cpp_param_string
        self.is_vector = "std::vector" in cpp_param_string

        remove_list = ["std::optional<", "std::vector<", ">", "=", "&"]
        for it in remove_list:
            cpp_param_string = cpp_param_string.replace(it, "")
        cpp_param_string = cpp_param_string.strip().split()
        remove_list = ("const", "constexpr")
        cpp_param_string = [it for it in cpp_param_string if it not in remove_list]
        assert len(cpp_param_string) <= 3

        self.param_type = cpp_param_string[0]
        self.param_name = cpp_param_string[1]
        self.default_value = None
        if len(cpp_param_string) >= 3:
            self.default_value = cpp_param_string[2]

        if self.param_type == "TensorArray":
            self.is_vector = True
            self.param_type = "Tensor"

    def get_signature_str(self):
        param_type_str = self.param_type
        if self.is_optional and self.is_vector:
            param_type_str = f"std::optional<std::vector<{self.param_type}>>"
        elif self.is_optional:
            param_type_str = f"std::optional<{self.param_type}>"
        elif self.is_vector:
            param_type_str = f"std::vector<{self.param_type}>"

        default_value_str = ""
        if self.default_value:
            default_value_str = f" = {self.default_value}"
        default_value_str = default_value_str.replace('"', '\\"')
        return f"{param_type_str} {to_lower_case_name(self.param_name)}{default_value_str}"

    def get_unpack_str(self, index):
        param_type_str = self.param_type

        if self.is_optional and self.is_vector:
            param_type_str = f"std::optional<std::vector<{self.param_type}>>"
        elif self.is_optional:
            param_type_str = f"std::optional<{self.param_type}>"
        elif self.is_vector:
            param_type_str = f"std::vector<{self.param_type}>"

        if self.default_value:
            return f"parserResult.UnpackOr<{param_type_str}>({index}, {self.default_value})"
        else:
            return f"parserResult.Unpack<{param_type_str}>({index})"

    def get_lambda_param_str(self):
        """Generate lambda parameter declaration, e.g. 'const std::optional<Tensor>& scale'."""
        param_type_str = self.param_type

        # Build wrapped type: outermost must be built first to avoid double-wrapping
        if self.is_optional and self.is_vector:
            param_type_str = f"std::optional<std::vector<{param_type_str}>>"
        elif self.is_optional:
            param_type_str = f"std::optional<{param_type_str}>"
        elif self.is_vector:
            param_type_str = f"std::vector<{param_type_str}>"

        name = to_lower_case_name(self.param_name)

        # Primitives: pass by value
        if self.param_type in ("int64_t", "int32_t", "double", "float", "bool"):
            return f"{param_type_str} {name}"
        # String: const ref
        if self.param_type == "std::string":
            return f"const {param_type_str}& {name}"
        # Tensor, Graph, Scalar, IntOrIntArray, Shape, DeviceMesh, PlacementSeq, etc.: const ref
        return f"const {param_type_str}& {name}"

    def get_arg_spec_str(self):
        """Generate nanobind arg spec, e.g. 'nb::arg("scale") = std::nullopt'."""
        name = to_lower_case_name(self.param_name)
        if self.default_value is not None:
            dv = self.default_value
            # Handle empty initializer list default for IntOrIntArray
            if dv == "{}" and self.param_type == "IntOrIntArray":
                dv = "IntOrIntArray()"
            return f'nb::arg("{name}") = {dv}'
        else:
            return f'nb::arg("{name}")'

    def get_call_arg_str(self):
        """Return just the snake_case parameter name for passing to the C++ function."""
        return to_lower_case_name(self.param_name)


class FuncSignature:
    def __init__(self, cpp_func_string: str):
        cpp_func_string = cpp_func_string.replace("const ", "")
        cpp_func_string = cpp_func_string.replace("constexpr ", "")
        cpp_func_string = cpp_func_string.replace("&", "")

        return_type_match = re.search(r"([\w<>:]+)\s+(\w+)\s*\(", cpp_func_string)
        assert return_type_match, f"Parse function name error: {cpp_func_string}"
        if return_type_match:
            self.return_type = return_type_match.group(1)
            self.function_name = return_type_match.group(2)

        parameters_match = re.search(r"\((.*?)\);", cpp_func_string)
        assert parameters_match, f"Parse parameter error: {cpp_func_string}"
        if parameters_match:
            param_strings = parameters_match.group(1).split(",")
            self.params = [FuncParam(param_string) for param_string in param_strings]

    def get_signature_str(self):
        param_string = [param.get_signature_str() for param in self.params]
        param_string = ", ".join(param_string)
        result = f"{self.return_type} {self.function_name}({param_string});"
        return result

    def get_param_get_unpack_str(self):
        param_string = [param.get_unpack_str(index) for index, param in enumerate(self.params)]
        return ", ".join(param_string)

    def get_python_function_name(self):
        return to_lower_case_name(self.function_name)

    def get_return_type_str(self):
        """Return type for lambda, mapping TensorArray -> std::vector<Tensor>."""
        if self.return_type == "TensorArray":
            return "std::vector<Tensor>"
        return self.return_type

    def get_lambda_params_str(self):
        """Comma-separated lambda parameter declarations."""
        param_strings = [param.get_lambda_param_str() for param in self.params]
        return ", ".join(param_strings)

    def get_arg_specs_str(self):
        """Comma-separated nanobind arg specs."""
        return ", ".join(param.get_arg_spec_str() for param in self.params)

    def get_call_args_str(self):
        """Comma-separated call argument names."""
        return ", ".join(param.get_call_arg_str() for param in self.params)


class Generator:
    def get_all_signature_string(self, dir, dest_path):
        all_header_files = glob.glob(os.path.join(dir, "*.h"))
        all_header_files = [file_name for file_name in all_header_files if "functional_option.h" not in file_name]
        assert len(all_header_files) > 0

        # Check need to regenereate file by modified time
        need_to_regenerate = False
        current_time_seconds = time.time()

        if os.path.exists(dest_path):
            dest_path_modify_time_seconds = os.path.getmtime(dest_path)
            if dest_path_modify_time_seconds < os.path.getmtime(os.path.abspath(__file__)):
                need_to_regenerate = True
            elif current_time_seconds < dest_path_modify_time_seconds:
                need_to_regenerate = True
            else:
                for header_file in all_header_files:
                    header_file_modify_time_seconds = os.path.getmtime(header_file)
                    if header_file_modify_time_seconds > dest_path_modify_time_seconds:
                        need_to_regenerate = True
                        break
        else:
            need_to_regenerate = True

        # Get signature_strings
        signature_strings = []
        for header_file in all_header_files:
            with open(header_file, "r") as file:
                file_content = file.read()

                left_string = "namespace functional {"
                right_string = "}  // namespace functional"
                start_index = file_content.find(left_string)
                end_index = file_content.find(right_string)
                assert start_index > 0 and end_index > start_index
                file_content = file_content[start_index + len(left_string) : end_index]

                def remove_cpp_comments(code_string):
                    code_string = re.sub(r"/\*.*?\*/", "", code_string, flags=re.DOTALL)
                    code_string = re.sub(r"//.*", "", code_string)
                    return code_string

                file_content = remove_cpp_comments(file_content)

                content = file_content.split(";")
                content = [" ".join(text.split()) for text in content]
                content = [text + ";" for text in content if len(text) > 0]

                signature_strings.extend(content)

        return signature_strings, need_to_regenerate

    def construct_func_signature(self, signature_strings, skip_func_names):
        result = {}
        assert len(signature_strings) > 0
        for signature_string in signature_strings:
            signature = FuncSignature(signature_string)
            if signature.function_name in skip_func_names:
                continue

            if signature.function_name not in result:
                result[signature.function_name] = []
            result[signature.function_name].append(signature)

        return result

    def generate_def_for_signatures(self, signatures_maps):
        result = []
        for function_name, signature_list in sorted(signatures_maps.items()):
            assert len(signature_list) > 0
            python_function_name = signature_list[0].get_python_function_name()

            for sig in signature_list:
                return_type = sig.get_return_type_str()
                lambda_params = sig.get_lambda_params_str()
                call_args = sig.get_call_args_str()
                arg_specs = sig.get_arg_specs_str()

                if return_type == "void":
                    call_statement = f"api::cpp::functional::{function_name}({call_args});"
                    def_string = f"""
        m.def("{python_function_name}", []({lambda_params}) -> void {{
            {call_statement}
        }}, {arg_specs});
"""
                elif return_type == "std::vector<Tensor>":
                    call_statement = f"auto result = api::cpp::functional::{function_name}({call_args});"
                    return_statement = "return dtorch::api::py::WrapTensorArray(result);"
                    def_string = f"""
        m.def("{python_function_name}", []({lambda_params}) -> nb::object {{
            {call_statement}
            {return_statement}
        }}, {arg_specs});
"""
                elif return_type == "Tensor":
                    call_statement = f"auto result = api::cpp::functional::{function_name}({call_args});"
                    return_statement = "return dtorch::api::py::WrapTensor(result);"
                    def_string = f"""
        m.def("{python_function_name}", []({lambda_params}) -> nb::object {{
            {call_statement}
            {return_statement}
        }}, {arg_specs});
"""
                else:
                    call_statement = f"return api::cpp::functional::{function_name}({call_args});"
                    def_string = f"""
        m.def("{python_function_name}", []({lambda_params}) -> {return_type} {{
            {call_statement}
        }}, {arg_specs});
"""

                result.append(def_string)
        return result

    def write_file(self, dest_path, def_for_signatures):
        dirname = os.path.dirname(dest_path)
        os.makedirs(dirname, exist_ok=True)

        with open(dest_path, "w") as file:
            file.write(file_begin)
            for def_for_signature in def_for_signatures:
                file.write(def_for_signature)
            file.write(file_end)

    def run(self, cpp_header_dir, dest_path, skip_func_names):
        signature_strings, need_to_regenerate = self.get_all_signature_string(cpp_header_dir, dest_path)
        if need_to_regenerate:
            signatures_maps = self.construct_func_signature(signature_strings, skip_func_names)
            def_for_signatures = self.generate_def_for_signatures(signatures_maps)
            self.write_file(dest_path, def_for_signatures)


def main():
    if len(sys.argv) < 3:
        print("Need more arguments, eg: python generate_py_bind_functional.py ./ ./build-linux")
        sys.exit(1)

    project_root = sys.argv[1]
    project_binary_root = sys.argv[2]
    cpp_header_dir = f"{project_root}/dtorch/api/cpp/functional/"
    dest_path = f"{project_binary_root}/dtorch/api/python/functional/py_bind_functional_generated.h"
    skip_func_names = {}  # "Zeros"

    # Generate file
    generator = Generator()
    generator.run(cpp_header_dir, dest_path, skip_func_names)


if __name__ == "__main__":
    sys.exit(main())
