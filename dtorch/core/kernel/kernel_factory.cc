/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "kernel_factory.h"

#include <memory>
#include <sstream>

#include "dtorch/common/debug.h"
#include "kernel_implement/convert_kernel.h"
#include "kernel_implement/copy_kernel.h"
#include "kernel_implement/create_kernel.h"
#include "kernel_implement/memory_kernel.h"
#include "kernel_implement/reduce_kernel.h"
#include "kernel_implement/sync_kernel.h"
#include "kernel_implement/view_kernel.h"

namespace dtorch {
namespace core {

template <typename KernelClass>
void KernelFactory::RegisterKernelConstructor(OperatorType opType) {
    auto kernelConstructorFunc = [](const KernelCreateCtx& ctx) -> std::unique_ptr<Kernel> {
        return std::make_unique<KernelClass>(ctx);
    };

    if (mKernelConstructorMap.count(opType) > 0) {
        DLogFatal() << "Can't RegisterKernelConstructor at same operator type: " << OpTypeToString(opType) << "."
                    << "Please check OperatorType mapping.";
    }

    mKernelConstructorMap[opType] = kernelConstructorFunc;
}

KernelFactory::KernelFactory() : mKernelConstructorMap() {
    RegisterKernelConstructor<ConvertKernel>(OperatorType::kConvert);
    RegisterKernelConstructor<CreateKernel>(OperatorType::kCreate);
    RegisterKernelConstructor<CopyKernel>(OperatorType::kCopy);
    RegisterKernelConstructor<ViewKernel>(OperatorType::kView);
    RegisterKernelConstructor<ViewKernel>(OperatorType::kReshape);
    RegisterKernelConstructor<ReduceKernel>(OperatorType::kReduce);
    RegisterKernelConstructor<MemoryKernel>(OperatorType::kMemory);
    RegisterKernelConstructor<SyncKernel>(OperatorType::kSync);
}

std::unique_ptr<Kernel> KernelFactory::ConstructKernel(const KernelCreateCtx& ctx) {
    OperatorType opType = ctx.op->GetOpType();
    const auto& iterator = mKernelConstructorMap.find(opType);

    if (iterator == mKernelConstructorMap.end()) {
        return std::make_unique<Kernel>(ctx);
    }

    return iterator->second(ctx);
}

std::unique_ptr<Kernel> KernelFactory::NewKernel(const KernelCreateCtx& ctx) { return ConstructKernel(ctx); }

}  // namespace core
}  // namespace dtorch
