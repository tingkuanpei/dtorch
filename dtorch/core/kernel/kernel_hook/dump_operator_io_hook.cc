
/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "dump_operator_io_hook.h"

#include <sstream>

#include "../kernel.h"
#include "dtorch/common/logging.h"
#include "dtorch/core/operators/operator.h"

namespace dtorch {
namespace core {

DumpOperatorIOHook::DumpOperatorIOHook(const std::string& path) : mOpIndex(0), mPath(), mMutex() {
    mPath = "./" + path;
    DLogInfo() << "Enable dump operator io tensor, dump operator folder is: " << mPath;
    DUnimplemented();
    // if (!filesystem::Exists(mPath)) {
    //     filesystem::CreateDir(mPath);
    //     DLogInfo() << "Create director: " << mPath;
    // }
}

void DumpOperatorIOHook::BeforeCompute(const Kernel& kernel) {
    std::lock_guard<std::mutex> guard(mMutex);
    IgnoreUnused(kernel);

    // const KernelExecuteCtx* kernelExecuteCtx = kernel.GetKernelExecuteCtx();
    // kernelExecuteCtx->stream->NativeSynchronize();

    // for (size_t i = 0; i < kernelExecuteCtx->inputs.size(); i++) {
    //     Tensor* inputTensor = kernelExecuteCtx->inputs[i];
    //     std::string fileName = GenerateTensorNpyFileName(kernelExecuteCtx->op, true, i);
    //     DLogInfo() << "dump tensor: " << fileName;
    //     DAlwaysAssert(TensorFunc::SaveAsNpyFormatFile(*inputTensor, fileName));
    // }
}

void DumpOperatorIOHook::AfterCompute(const Kernel& kernel) {
    std::lock_guard<std::mutex> guard(mMutex);
    IgnoreUnused(kernel);

    // const KernelExecuteCtx* kernelExecuteCtx = kernel.GetKernelExecuteCtx();
    // kernelExecuteCtx->stream->NativeSynchronize();

    // for (size_t i = 0; i < kernelExecuteCtx->outputs.size(); i++) {
    //     Tensor* outputTensor = kernelExecuteCtx->outputs[i];
    //     std::string fileName = GenerateTensorNpyFileName(kernelExecuteCtx->op, false, i);
    //     DLogInfo() << "dump tensor: " << fileName;
    //     DAlwaysAssert(TensorFunc::SaveAsNpyFormatFile(*outputTensor, fileName));
    // }

    // for (size_t i = 0; i < kernelExecuteCtx->inputs.size(); i++) {
    //     Tensor* inputTensor = kernelExecuteCtx->inputs[i];
    //     Operand* inputOperand = kernelExecuteCtx->op->GetInputOperand(i);
    //     if (inputOperand->GetModifyConsumerOps().count(kernelExecuteCtx->op) == 0) {
    //         continue;
    //     }

    //     std::string fileName = GenerateTensorNpyFileName(kernelExecuteCtx->op, true, i, true);
    //     DLogInfo() << "dump tensor: " << fileName;
    //     DAlwaysAssert(TensorFunc::SaveAsNpyFormatFile(*inputTensor, fileName));
    // }

    mOpIndex++;
}

std::string DumpOperatorIOHook::GenerateTensorNpyFileName(const Operator* op, bool isInput, size_t index,
                                                          bool isModifyedInput) {
    std::stringstream ss;
    ss << mPath << "/operator[" << mOpIndex << "]_(" << OpTypeToString(op->GetOpType()) << "_" << op->GetOpName()
       << ")_";
    if (isInput && !isModifyedInput) {
        ss << "input[" << index << "]";
    } else if (isInput && !isModifyedInput) {
        ss << "modifyed_input[" << index << "]";
    } else {
        ss << "output[" << index << "]";
    }

    ss << ".npy";
    return ss.str();
}

}  // namespace core
}  // namespace dtorch
