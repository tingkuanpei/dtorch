/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "torch_util.h"

#include <cstdint>

#include <torch/torch.h>

#include "dtorch/api/cpp/data_kind.h"
#include "dtorch/api/cpp/device.h"
#include "dtorch/common/config.h"
#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/common/string.h"
#include "dtorch/core/type.h"
#include "dtorch/external/boost/boost_serialization.h"
#include "dtorch/external/boost/boost_serialization_torch.h"
#include "dtorch/external/cuda/cuda_device.h"
#include "dtorch/external/cuda/cuda_error.h"
#include "dtorch/external/cuda/cuda_util.h"

#if DTORCH_WITH_CUDA
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace dtorch {
namespace external {
namespace torch {

size_t TorchUtil::CudaDeviceCount() { return ::torch::cuda::device_count(); }

Device TorchUtil::ToDevice(const ::torch::Device& torchDevice) {
    Device device;
    if (torchDevice.type() == ::torch::DeviceType::CUDA) {
        device.deviceKind = DeviceKind::kGpu;
        device.deviceId = torchDevice.index() < 0 ? 0 : torchDevice.index();
    }
    return device;
}

::torch::Device TorchUtil::ToDevice(const Device& device) {
    ::torch::DeviceType type = ::torch::DeviceType::CPU;
    ::torch::DeviceIndex index = -1;

    if (device.deviceKind == DeviceKind::kGpu) {
        type = ::torch::DeviceType::CUDA;
        index = device.deviceId;
    }

    return ::torch::Device(type, index);
}

DataKind TorchUtil::ToDataKind(const ::torch::ScalarType& torchScalarType) {
    if (torchScalarType == ::torch::kFloat16) {
        return DataKind::kFloat16;
    } else if (torchScalarType == ::torch::kBFloat16) {
        return DataKind::kBFloat16;
    } else if (torchScalarType == ::torch::kFloat32) {
        return DataKind::kFloat32;
    } else if (torchScalarType == ::torch::kDouble) {
        return DataKind::kFloat64;
    } else if (torchScalarType == ::torch::kLong) {
        return DataKind::kInt64;
    } else if (torchScalarType == ::torch::kInt) {
        return DataKind::kInt32;
    } else if (torchScalarType == ::torch::kShort) {
        return DataKind::kInt16;
    } else if (torchScalarType == ::torch::kChar) {
        return DataKind::kInt8;
    } else if (torchScalarType == ::torch::kByte) {
        return DataKind::kUInt8;
#if !DTORCH_INTEL_MAXOS_TORCH_2_2_2
    } else if (torchScalarType == ::torch::kUInt16) {
        return DataKind::kUInt16;
    } else if (torchScalarType == ::torch::kUInt32) {
        return DataKind::kUInt32;
    } else if (torchScalarType == ::torch::kUInt64) {
        return DataKind::kUInt64;
#endif
    } else if (torchScalarType == ::torch::kBool) {
        return DataKind::kBool;
    }

    DLogError() << "Unsupport dtype: " << torchScalarType;
    DUnimplemented();
    return DataKind::kFloat32;
}

::torch::ScalarType TorchUtil::ToScalarType(DataKind dataKind) {
    switch (dataKind) {
        case DataKind::kFloat16:
            return ::torch::ScalarType::Half;
        case DataKind::kFloat32:
            return ::torch::ScalarType::Float;
        case DataKind::kFloat64:
            return ::torch::ScalarType::Double;
        case DataKind::kBFloat16:
            return ::torch::ScalarType::BFloat16;
        case DataKind::kInt8:
            return ::torch::ScalarType::Char;
        case DataKind::kInt16:
            return ::torch::ScalarType::Short;
        case DataKind::kInt32:
            return ::torch::ScalarType::Int;
        case DataKind::kInt64:
            return ::torch::ScalarType::Long;
        case DataKind::kUInt8:
            return ::torch::ScalarType::Byte;
#if !DTORCH_INTEL_MAXOS_TORCH_2_2_2
        case DataKind::kUInt16:
            return ::torch::ScalarType::UInt16;
        case DataKind::kUInt32:
            return ::torch::ScalarType::UInt32;
        case DataKind::kUInt64:
            return ::torch::ScalarType::UInt64;
#endif
        case DataKind::kBool:
            return ::torch::ScalarType::Bool;
        default:
            break;
    }

    DUnimplemented();
    return ::torch::ScalarType::Float;
}

c10::Scalar TorchUtil::ToScalar(const Scalar& scalar) {
    if (scalar.IsFloatingPoint()) {
        return c10::Scalar(scalar.Value<double>());
    } else if (scalar.IsSigned()) {
        return c10::Scalar(scalar.Value<int64_t>());
    } else {
        return c10::Scalar(static_cast<int64_t>(scalar.Value<uint64_t>()));
    }
}

Shape TorchUtil::GetShape(const ::torch::Tensor& torchTensor) { return Shape(torchTensor.sizes().vec()); }

Stride TorchUtil::GetStride(const ::torch::Tensor& torchTensor) { return Stride(torchTensor.strides().vec()); }

DataKind TorchUtil::GetDataKind(const ::torch::Tensor& torchTensor) {
    // const static std::unordered_map<caffe2::TypeMeta, DataKind> kDataKindMap = {
    //     {::torch::kFloat16, DataKind::kFloat16}, {::torch::kBFloat16, DataKind::kBFloat16},
    //     {::torch::kFloat32, DataKind::kFloat32}, {::torch::kDouble, DataKind::kFloat64},
    //     {::torch::kLong, DataKind::kInt64},      {::torch::kInt, DataKind::kInt32},
    //     {::torch::kShort, DataKind::kInt16},     {::torch::kChar, DataKind::kInt8},
    //     {::torch::kByte, DataKind::kUInt8},      {::torch::kUInt16, DataKind::kUInt16},
    //     {::torch::kUInt32, DataKind::kUInt32},   {::torch::kUInt64, DataKind::kUInt64},
    //     {::torch::kBool, DataKind::kBool}};
    // DDebugAssert(static_cast<int>(kDataKindMap.size()) == EnumAsInteger<DataKind>(DataKind::kCount));

    // DDebugAssert(kDataKindMap.count(torchTensor.dtype()));
    // return kDataKindMap[torchTensor.dtype()];

    // TODO: fix cannot convert ‘caffe2::TypeMeta’ to ‘const c10::ScalarType&’
    // return ToDataKind(torchTensor.dtype());

    if (torchTensor.dtype() == ::torch::kFloat16) {
        return DataKind::kFloat16;
    } else if (torchTensor.dtype() == ::torch::kBFloat16) {
        return DataKind::kBFloat16;
    } else if (torchTensor.dtype() == ::torch::kFloat32) {
        return DataKind::kFloat32;
    } else if (torchTensor.dtype() == ::torch::kDouble) {
        return DataKind::kFloat64;
    } else if (torchTensor.dtype() == ::torch::kLong) {
        return DataKind::kInt64;
    } else if (torchTensor.dtype() == ::torch::kInt) {
        return DataKind::kInt32;
    } else if (torchTensor.dtype() == ::torch::kShort) {
        return DataKind::kInt16;
    } else if (torchTensor.dtype() == ::torch::kChar) {
        return DataKind::kInt8;
    } else if (torchTensor.dtype() == ::torch::kByte) {
        return DataKind::kUInt8;
#if !DTORCH_INTEL_MAXOS_TORCH_2_2_2
    } else if (torchTensor.dtype() == ::torch::kUInt16) {
        return DataKind::kUInt16;
    } else if (torchTensor.dtype() == ::torch::kUInt32) {
        return DataKind::kUInt32;
    } else if (torchTensor.dtype() == ::torch::kUInt64) {
        return DataKind::kUInt64;
#endif
    } else if (torchTensor.dtype() == ::torch::kBool) {
        return DataKind::kBool;
    }

    DLogError() << "Unsupport dtype: " << torchTensor.dtype();
    DUnimplemented();
    return DataKind::kFloat32;
}

Device TorchUtil::GetDevice(const ::torch::Tensor& torchTensor) { return ToDevice(torchTensor.device()); }

std::vector<int64_t> TorchUtil::ToInt64Vec(const ::torch::Tensor& torchTensor) {
    ::torch::Tensor torchTensorContiguous = torchTensor.contiguous();
    DDebugAssert(torchTensorContiguous.device().is_cpu());
    DDebugAssert(torchTensorContiguous.dtype() == ::torch::kInt64);

    std::vector<int64_t> result(torchTensorContiguous.numel());
    std::memcpy(result.data(), torchTensorContiguous.data_ptr<int64_t>(),
                torchTensorContiguous.numel() * sizeof(int64_t));
    return result;
}

std::unordered_set<int64_t> TorchUtil::ToInt64Set(const ::torch::Tensor& torchTensor) {
    std::vector<int64_t> vec = TorchUtil::ToInt64Vec(torchTensor);
    return std::unordered_set<int64_t>(vec.begin(), vec.end());
}

::torch::indexing::Slice TorchUtil::ToSlice(const Slice& slice) {
    auto ToTorchOptional = [](std::optional<int64_t> value) {
        if (value.has_value()) {
            return std::optional<c10::SymInt>(value.value());
        } else {
            return std::optional<c10::SymInt>(std::nullopt);
        }
    };

    return ::torch::indexing::Slice(ToTorchOptional(slice.start), ToTorchOptional(slice.stop),
                                    ToTorchOptional(slice.step));
}

::torch::indexing::TensorIndex TorchUtil::ToIndex(const Index& index) {
    if (index.IsEllipsis()) {
        return ::torch::indexing::TensorIndex(at::indexing::Ellipsis);
    } else if (index.IsNone()) {
        return ::torch::indexing::TensorIndex(at::indexing::None);
    } else if (index.IsInteger()) {
        return ::torch::indexing::TensorIndex(index.GetInteger());
    } else if (index.IsSlice()) {
        return ::torch::indexing::TensorIndex(ToSlice(index.GetSlice()));
    } else if (index.IsTensor()) {
        return ::torch::indexing::TensorIndex(index.GetTensor());
    } else {
        DUnimplemented();
        return ::torch::indexing::TensorIndex(0);
    }
}

std::vector<::torch::indexing::TensorIndex> TorchUtil::ToIndex(const std::vector<Index>& indexs) {
    std::vector<::torch::indexing::TensorIndex> result;
    for (auto& it : indexs) {
        result.push_back(ToIndex(it));
    }
    return result;
}

::torch::Tensor TorchUtil::CreateTensor(const Shape& shape, const Device& device, DataKind dataKind,
                                        const std::vector<char>& dataBuffer) {
    ::torch::Device torchDevice = TorchUtil::ToDevice(device);
    ::torch::Device cpuDevice = TorchUtil::ToDevice(Device::GetDefaultCpuDevice());
    ::torch::ScalarType scalarType = TorchUtil::ToScalarType(dataKind);

    auto options = ::torch::TensorOptions().dtype(scalarType).device(cpuDevice);
    ::torch::Tensor tensor = ::torch::empty(shape.Vec(), options);
    DAlwaysAssert(tensor.nbytes() == dataBuffer.size());
    std::memcpy(tensor.data_ptr(), dataBuffer.data(), tensor.nbytes());
    tensor = tensor.to(torchDevice);
    return tensor;
}

std::vector<char> TorchUtil::ToCharVec(const ::torch::Tensor& torchTensor) {
    ::torch::Tensor torchTensorContiguous = torchTensor.contiguous().cpu();
    DDebugAssert(torchTensorContiguous.device().is_cpu());

    std::vector<char> result(torchTensorContiguous.nbytes());
    std::memcpy(result.data(), torchTensorContiguous.data_ptr(), torchTensorContiguous.nbytes());
    return result;
}

std::string TorchUtil::ToIpcMemHandle(const ::torch::Tensor& torchTensor) {
    DAlwaysAssert(cuda::CudaDevice::IsSupportIpc(GetDevice(torchTensor).deviceId));

    std::stringstream ss(std::ios::out | std::ios::binary);
    boost::BinaryOArchive boa(ss);
    DeviceKind deviceKind = TorchUtil::GetDevice(torchTensor).deviceKind;
    bool isEmpty = torchTensor.nbytes() == 0;
    boa << deviceKind;
    boa << isEmpty;

    if (!isEmpty && deviceKind == DeviceKind::kGpu) {
        boa << GetDevice(torchTensor);
        boa << GetShape(torchTensor);
        boa << GetStride(torchTensor);
        boa << GetDataKind(torchTensor);

        cudaIpcMemHandle_t memHandle;
        DCudaCheckError(cudaIpcGetMemHandle(&memHandle, torchTensor.data_ptr()));
        boa << ::boost::serialization::make_array(memHandle.reserved, CUDA_IPC_HANDLE_SIZE);

        size_t storageOffset = cuda::CudaUtil::GetMemOffeset(torchTensor.data_ptr());
        boa << storageOffset;
    } else if (isEmpty || deviceKind == DeviceKind::kCpu) {
        boa << torchTensor;
    } else {
        DLogFatal() << "TorchUtil::ToIpcMemHandle Only support gpu and cpu";
        DUnimplemented();
    }

    return ss.str();
}

::torch::Tensor TorchUtil::FromIpcMemHandle(const std::string& str) {
    std::stringstream ss(str, std::ios::in | std::ios::binary);
    boost::BinaryIArchive bia(ss);
    DeviceKind deviceKind;
    bool isEmpty;
    bia >> deviceKind;
    bia >> isEmpty;

    if (!isEmpty && deviceKind == DeviceKind::kGpu) {
        Device device;
        Shape shape;
        Stride stride;
        DataKind datakKind;
        cudaIpcMemHandle_t memHandle;
        size_t storageOffset;
        bia >> device;
        bia >> shape;
        bia >> stride;
        bia >> datakKind;
        bia >> ::boost::serialization::make_array(memHandle.reserved, CUDA_IPC_HANDLE_SIZE);
        bia >> storageOffset;

        cuda::CudaDeviceGuard guard(device.deviceId);
        void* ptr = nullptr;
        // Call cudaIpcGetMemHandle() then call cudaIpcOpenMemHandle() to get the cuda ipc memory in same process,
        // will raise error: "CUDA error name: cudaErrorDeviceUninitialized. CUDA error string: invalid device context".
        // This also can be reproduced by modifying simpleIPC.cu in cuda-samples.
        //
        // Reference:
        // https://stackoverflow.com/questions/32107137/invalid-device-ordinal-cudaerrorinvaliddevice-returned-on-cudaipcopenmemhand

        // Calling cudaFree on an exported memory region before calling cudaIpcCloseMemHandle in the importing context
        // will result in undefined behavior.
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g01050a29fefde385b1042081ada4cde9
        DCudaCheckError(cudaIpcOpenMemHandle(&ptr, memHandle, cudaIpcMemLazyEnablePeerAccess));
        ::torch::Tensor result = ::torch::from_blob(
            static_cast<char*>(ptr) + storageOffset, shape.Vec(), stride.Vec(),
            [=](void* data) { DCudaCheckError(cudaIpcCloseMemHandle(static_cast<char*>(data) - storageOffset)); },
            at::TensorOptions().dtype(ToScalarType(datakKind)).device(ToDevice(device)));
        return result;
    } else if (isEmpty || deviceKind == DeviceKind::kCpu) {
        ::torch::Tensor result;
        bia >> result;
        return result;
    } else {
        DLogFatal() << "TorchUtil::FromIpcMemHandle Only support gpu and cpu";
        DUnimplemented();
        return ::torch::Tensor();
    }
}

std::shared_ptr<::torch::Generator> TorchUtil::GetGenerator(const Device& device) {
    std::shared_ptr<::torch::Generator> result;
    if (device.deviceKind == DeviceKind::kCpu) {
        result = std::make_shared<::torch::Generator>(at::make_intrusive<::torch::CPUGeneratorImpl>());
    }
#if DTORCH_WITH_CUDA
    else if (device.deviceKind == DeviceKind::kGpu) {
        result = std::make_shared<::torch::Generator>(
            at::make_intrusive<at::CUDAGeneratorImpl>(static_cast<::torch::DeviceIndex>(device.deviceId)));
    }
#endif
    else {
        DLogFatal() << "GetGenerator unsupported device: " << device;
        DUnimplemented();
    }
    return result;
}

std::string CUDAStreamToString(const c10::cuda::CUDAStream& stream) {
    std::stringstream ss;
    ss << "CUDAStream(device_id=" << stream.device_index() << ", stream_id=" << stream.id() << ")";
    return ss.str();
}

std::ostream& operator<<(std::ostream& os, const c10::cuda::CUDAStream& stream) {
    os << CUDAStreamToString(stream);
    return os;
}

}  // namespace torch
}  // namespace external
}  // namespace dtorch
