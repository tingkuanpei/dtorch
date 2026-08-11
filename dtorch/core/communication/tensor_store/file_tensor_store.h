/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "tensor_store.h"

namespace dtorch {
namespace core {
namespace communication {

class FileTensorStore : public TensorStore {
public:
    FileTensorStore(const std::string& fileName, int worldSize);

    ~FileTensorStore();

    void SrcSet(const std::string& key, const torch::Tensor& value, DeviceStream& stream, size_t getCount,
                std::optional<DeviceKind> destGetDeviceKind = std::nullopt) override;

    void SrcWaitUntilGetFinished(const std::string& key, DeviceStream& stream) override;

    torch::Tensor DestGet(const std::string& key, DeviceStream& stream) override;

    void DestFinishGet(const std::string& key, DeviceStream& stream) override;

    void Reset() override;

    void Barrier() override;

private:
    void SetLongString(const std::string& key, const std::string& value);

    std::string GetLongString(const std::string& key) const;

private:
    int64_t mResetCount;
    std::string mResetBaseFileName;
    struct Impl;
    mutable std::shared_ptr<Impl> mImplPtr;
};

}  // namespace communication
}  // namespace core
}  // namespace dtorch
