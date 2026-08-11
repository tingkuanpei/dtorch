/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <torch/torch.h>

#include "dtorch/api/cpp/generator.h"
#include "dtorch/common/logging.h"
#include "dtorch/external/boost/boost_serialization.h"
#include "dtorch/external/torch/torch_util.h"

namespace boost {
namespace serialization {

template <typename Archive>
void serialize(Archive& ar, ::torch::Tensor& tensor, const unsigned int /*version*/) {
    using dtorch::api::cpp::DataKind;
    using dtorch::api::cpp::Device;
    using dtorch::api::cpp::Shape;
    using dtorch::external::torch::TorchUtil;

    bool defined = tensor.defined();
    ar & defined;

    if (defined) {
        Shape shape;
        Device device;
        DataKind dataKind;
        std::vector<char> dataBuffer;

        if constexpr (Archive::is_saving::value) {
            shape = TorchUtil::GetShape(tensor);
            device = TorchUtil::GetDevice(tensor);
            dataKind = TorchUtil::GetDataKind(tensor);
            dataBuffer = TorchUtil::ToCharVec(tensor);
        }

        ar & shape;
        ar & device;
        ar & dataKind;
        ar & dataBuffer;

        if constexpr (Archive::is_loading::value) {
            tensor = TorchUtil::CreateTensor(shape, device, dataKind, dataBuffer);
        }
    } else {
        if constexpr (Archive::is_loading::value) {
            tensor = ::torch::Tensor();
        }
    }
}

template <typename Archive>
void serialize(Archive& ar, ::torch::Generator& generator, const unsigned int /*version*/) {
    using dtorch::api::cpp::Device;
    using dtorch::external::torch::TorchUtil;

    bool defined = generator.defined();
    ar & defined;

    if (defined) {
        Device device;
        ::torch::Tensor state;

        if constexpr (Archive::is_saving::value) {
            device = TorchUtil::ToDevice(generator.device());
            state = generator.get_state();
        }

        ar & device;
        ar & state;

        if constexpr (Archive::is_loading::value) {
            generator = *TorchUtil::GetGenerator(device);
            generator.set_state(state);
        }
    } else {
        generator = ::torch::Generator();
    }
}

template <class Archive>
void serialize(Archive& ar, dtorch::api::cpp::Generator& generator, const unsigned int /*version*/) {
    std::shared_ptr<torch::Generator> torchGenerator;
    if constexpr (Archive::is_saving::value) {
        torchGenerator = generator.GetSharedTorchGenerator();
        ar & torchGenerator;
    } else {
        ar & torchGenerator;
        generator = dtorch::api::cpp::Generator(torchGenerator);
    }
}

}  // namespace serialization
}  // namespace boost
