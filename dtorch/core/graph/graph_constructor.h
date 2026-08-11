/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>
#include <optional>

#include "dtorch/api/cpp/tensor.h"
#include "dtorch/core/graph/eager_graph_executor.h"

namespace dtorch {
namespace core {

using api::cpp::Graph;

class GraphConstructor;

class ApiTensorRefCountHolder {
public:
    ApiTensorRefCountHolder(GraphConstructor* graphConstructor, const Operand* operand);

    ~ApiTensorRefCountHolder();

    DTORCH_DISABLE_COPY_AND_MOVE(ApiTensorRefCountHolder);

private:
    GraphConstructor* mGraphConstructor;
    const Operand* mOperand;
};

class GraphConstructor {
public:
    // Helper function for construct graph from api::cpp::Tensor, inputs tensor can from different graph
    static api::cpp::Tensor AddOperator(std::unique_ptr<OpParam> opParamPtr, const api::cpp::TensorArray& inputs,
                                        std::optional<Graph> graph = std::nullopt);

    static api::cpp::TensorArray AddOperator(std::unique_ptr<OpParam> opParamPtr, const api::cpp::TensorArray& inputs,
                                             bool /*returnArray*/, std::optional<Graph> graph = std::nullopt);

public:
    GraphConstructor(const GraphOption& graphOption, const std::string& publisherAddress,
                     const std::string& pushPullAddress);

    ~GraphConstructor();

    api::cpp::TensorArray AddOperator(Graph& graph, std::unique_ptr<OpParam> opParamPtr,
                                      const api::cpp::TensorArray& inputs);

    void SendOperatorToExecutor(std::unique_ptr<Operator> op);

    // Forwarded to EagerGraphExecutor. Must be called after MainNode::CreateGraph so WorkerNodes
    // exist to report "ready".
    void WaitAllRunnerReady();

    void ApiTensorRefCountIncrease(const Operand* operand);

    void ApiTensorRefCountDecrease(const Operand* operand);

    DTORCH_FORCEINLINE bool CountOperand(const Operand* operand) const noexcept {
        return mOperandCaches.count(operand) > 0;
    }

    void SetGraphName(const std::string& name);

    DTORCH_FORCEINLINE const std::string& GetGraphName() { return mGraphName; }

    void SetOperandName(const Operand* operand, const std::string& name);

    const std::string& GetOperandName(const Operand* operand);

private:
    void AddApiTensorNoHoldOperand(const Operand* operand);

    void InitOperandCache(const Operand* operand);

protected:
    struct OperandCache {
        std::string name;
        // The GraphConstructor does not support multi-threaded calls, so there is no need to use atomic here
        int apiTensorRefCount;

        OperandCache() : name(), apiTensorRefCount(0) {}
    };
    std::unordered_map<const Operand*, OperandCache> mOperandCaches;
    std::string mGraphName;
    // Need to destructor under python gil scopedRelease, so use std::unique_ptr
    std::unique_ptr<EagerGraphExecutor> mEagerGraphExecutor;
    EGEMessageQueue& mEGEMessageQueue;
};

}  // namespace core
}  // namespace dtorch
