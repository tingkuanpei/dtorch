/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <functional>
#include <memory>
#include <unordered_map>

#include "dtorch/common/debug.h"
#include "dtorch/core/blob.h"
#include "dtorch/core/communication/thread_group/thread_group_manager.h"
#include "dtorch/core/operators/operator.h"
#include "dtorch/core/operators/operator_param.h"

namespace dtorch {
namespace core {

class Kernel;
class KernelStream;

struct KernelCreateCtx {
    std::shared_ptr<Operator> op;
    const std::vector<AllDeviceBlobs>& inputs;
    const std::vector<AllDeviceBlobs>& outputs;
    KernelStream* stream;
    Device globalDevice;
    Device localDevice;
    size_t numKernelForThisOp;
    communication::ThreadGroupManager* threadGroupManager;

public:
    KernelCreateCtx(const std::shared_ptr<Operator>& op, const std::vector<AllDeviceBlobs>& inputs,
                    const std::vector<AllDeviceBlobs>& outputs, KernelStream* stream, const Device& globalDevice,
                    const Device& localDevice, size_t numKernelForThisOp,
                    communication::ThreadGroupManager* threadGroupManager)
        : op(op),
          inputs(inputs),
          outputs(outputs),
          stream(stream),
          globalDevice(globalDevice),
          localDevice(localDevice),
          numKernelForThisOp(numKernelForThisOp),
          threadGroupManager(threadGroupManager) {}

    DTORCH_DISABLE_COPY_AND_MOVE(KernelCreateCtx);
};

class KernelFactory {
public:
    DTORCH_API_FORCEINLINE static KernelFactory& GetSingleton() {
        static KernelFactory singleton;
        return singleton;
    }

public:
    std::unique_ptr<Kernel> NewKernel(const KernelCreateCtx& ctx);

private:
    KernelFactory();

    template <typename KernelClass>
    void RegisterKernelConstructor(OperatorType opType);

    std::unique_ptr<Kernel> ConstructKernel(const KernelCreateCtx& ctx);

private:
    using KernelConstructorFunc = std::function<std::unique_ptr<Kernel>(const KernelCreateCtx&)>;

    std::unordered_map<OperatorType, KernelConstructorFunc> mKernelConstructorMap;
};

}  // namespace core
}  // namespace dtorch
