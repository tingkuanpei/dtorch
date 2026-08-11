/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "dtorch/external/torch/torch_util.h"
#include "eager_graph_executor.h"
#include "eager_graph_executor_message.h"

namespace dtorch {
namespace core {

struct AddOperatorEEMsg : public EGEMessage {
    std::unique_ptr<Operator> op;

public:
    AddOperatorEEMsg() : op() {}

    DTORCH_DISABLE_COPY_AND_MOVE(AddOperatorEEMsg);

    DTORCH_FORCEINLINE void ProcessEGEMessage(EagerGraphExecutor& eagerGraphExecutor) override {
        eagerGraphExecutor.AddOperator(std::move(op));
    }
};

struct ApiTensorNoHoldEEMsg : public EGEMessage {
    const Operand* operand;

public:
    ApiTensorNoHoldEEMsg() : operand(nullptr) {}

    DTORCH_DISABLE_COPY_AND_MOVE(ApiTensorNoHoldEEMsg);

    DTORCH_FORCEINLINE void ProcessEGEMessage(EagerGraphExecutor& eagerGraphExecutor) override {
        eagerGraphExecutor.AddApiTensorNoHoldOperand(operand);
    }
};

struct SetOperandNameEEMsg : public EGEMessage {
    const Operand* operand;
    std::string name;

public:
    SetOperandNameEEMsg() : operand(nullptr), name() {}

    DTORCH_DISABLE_COPY_AND_MOVE(SetOperandNameEEMsg);

    DTORCH_FORCEINLINE void ProcessEGEMessage(EagerGraphExecutor& eagerGraphExecutor) override {
        eagerGraphExecutor.SetOperandName(operand, name);
    }
};

struct SetGraphNameEEMsg : public EGEMessage {
    std::string name;

public:
    SetGraphNameEEMsg() : name() {}

    DTORCH_DISABLE_COPY_AND_MOVE(SetGraphNameEEMsg);

    DTORCH_FORCEINLINE void ProcessEGEMessage(EagerGraphExecutor& eagerGraphExecutor) override {
        eagerGraphExecutor.SetGraphName(name);
    }
};

}  // namespace core
}  // namespace dtorch
