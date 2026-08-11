/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "graph_constructor.h"

#include <torch/torch.h>

#include "dtorch/core/graph/eager_graph_executor_message_imp.h"
#include "dtorch/core/operators/operator_factory.h"
#include "dtorch/core/operators/operator_param.h"
#include "dtorch/external/python/python_gil.h"

namespace dtorch {
namespace core {

ApiTensorRefCountHolder::ApiTensorRefCountHolder(GraphConstructor* graphConstructor, const Operand* operand)
    : mGraphConstructor(graphConstructor), mOperand(operand) {
    mGraphConstructor->ApiTensorRefCountIncrease(mOperand);
}

ApiTensorRefCountHolder::~ApiTensorRefCountHolder() { mGraphConstructor->ApiTensorRefCountDecrease(mOperand); }

api::cpp::Tensor GraphConstructor::AddOperator(std::unique_ptr<OpParam> opParamPtr, const api::cpp::TensorArray& inputs,
                                               std::optional<Graph> graph) {
    auto tensorArray = GraphConstructor::AddOperator(std::move(opParamPtr), inputs, true, graph);
    DDebugAssert(tensorArray.size() == 1);
    return tensorArray[0];
}

api::cpp::TensorArray GraphConstructor::AddOperator(std::unique_ptr<OpParam> opParamPtr,
                                                    const api::cpp::TensorArray& inputs, bool /*returnArray*/,
                                                    std::optional<Graph> graph) {
    core::GraphConstructor* graphConstuctor = nullptr;
    if (graph.has_value()) {
        graphConstuctor = graph.value().GetGraphConstructor();
    } else {
        DAlwaysAssert(inputs.size() > 0);
        graph = inputs[0].GetGraph();
        graphConstuctor = inputs[0].GetGraph().GetGraphConstructor();
    }

    return graphConstuctor->AddOperator(graph.value(), std::move(opParamPtr), inputs);
}

GraphConstructor::GraphConstructor(const GraphOption& graphOption, const std::string& publisherAddress,
                                   const std::string& pushPullAddress)
    : mOperandCaches(),
      mGraphName(),
      mEagerGraphExecutor(std::make_unique<EagerGraphExecutor>(graphOption, publisherAddress, pushPullAddress)),
      mEGEMessageQueue(mEagerGraphExecutor->GetEGEMessageQueue()) {}

void GraphConstructor::WaitAllRunnerReady() { mEagerGraphExecutor->WaitAllRunnerReady(); }

GraphConstructor::~GraphConstructor() {
    // Manual call destructor under python gil scopedRelease
    auto scopedRelease = external::python::GetPythonGilScopedRelease();
    mEagerGraphExecutor.reset();
}

api::cpp::TensorArray GraphConstructor::AddOperator(Graph& graph, std::unique_ptr<OpParam> opParamPtr,
                                                    const api::cpp::TensorArray& inputs) {
    // 1. Check same graph
    for (auto& input : inputs) {
        if (input.GetGraph().GetGraphConstructor() != this) {
            std::stringstream ss;
            ss << "Input must in same graph, but get two: graph id("
               << reinterpret_cast<uint64_t>(input.GetGraph().GetGraphConstructor()) << ") vs id("
               << reinterpret_cast<uint64_t>(this) << ")";
            throw std::invalid_argument(ss.str());
        }
    }

    // 2. Construct operator
    OperandArray inputOperands(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
        inputOperands[i] = inputs[i].GetOperand();
    }
    std::unique_ptr<Operator> op =
        OperatorFactory::GetSingleton().NewOperatorOrThrow(std::move(opParamPtr), inputOperands);

    // 3. Check operator device id valid or not
    mEagerGraphExecutor->CheckSupportOrThrow(*op);

    // 4. send operator to executor
    OperandArray outputOperands = op->GetOutputOperands();
    SendOperatorToExecutor(std::move(op));

    // 5. Return operator output api::cpp::Tensor
    api::cpp::TensorArray tensorArray;
    for (auto output : outputOperands) {
        DDebugAssert(CountOperand(output.get()));
        tensorArray.emplace_back(graph, output);
    }
    return tensorArray;
}

void GraphConstructor::SendOperatorToExecutor(std::unique_ptr<Operator> op) {
    for (const auto& it : op->GetOutputOperands()) {
        InitOperandCache(it.get());
    }

    std::unique_ptr<AddOperatorEEMsg> message = std::make_unique<AddOperatorEEMsg>();
    message->op = std::move(op);
    mEGEMessageQueue.PushMessage(std::move(message));
}

void GraphConstructor::ApiTensorRefCountIncrease(const Operand* operand) {
    DDebugAssert(mOperandCaches.count(operand) > 0);
    mOperandCaches[operand].apiTensorRefCount++;
}

void GraphConstructor::ApiTensorRefCountDecrease(const Operand* operand) {
    auto it = mOperandCaches.find(operand);
    DDebugAssert(it != mOperandCaches.end());
    DDebugAssert(it->second.apiTensorRefCount >= 1);

    if (it->second.apiTensorRefCount == 1) {
        mOperandCaches.erase(it);
        this->AddApiTensorNoHoldOperand(operand);
    } else {
        it->second.apiTensorRefCount--;
    }
}

void GraphConstructor::AddApiTensorNoHoldOperand(const Operand* operand) {
    std::unique_ptr<ApiTensorNoHoldEEMsg> message = std::make_unique<ApiTensorNoHoldEEMsg>();
    message->operand = operand;
    mEGEMessageQueue.PushMessage(std::move(message));
}

void GraphConstructor::SetGraphName(const std::string& name) {
    mGraphName = name;

    std::unique_ptr<SetGraphNameEEMsg> message = std::make_unique<SetGraphNameEEMsg>();
    message->name = name;
    mEGEMessageQueue.PushMessage(std::move(message));
}

void GraphConstructor::SetOperandName(const Operand* operand, const std::string& name) {
    DDebugAssert(CountOperand(operand));
    mOperandCaches[operand].name = name;

    std::unique_ptr<SetOperandNameEEMsg> message = std::make_unique<SetOperandNameEEMsg>();
    message->operand = operand;
    message->name = name;
    mEGEMessageQueue.PushMessage(std::move(message));
}

const std::string& GraphConstructor::GetOperandName(const Operand* operand) {
    DDebugAssert(CountOperand(operand));
    return mOperandCaches[operand].name;
}

void GraphConstructor::InitOperandCache(const Operand* operand) {
    DDebugAssert(mOperandCaches.count(operand) == 0);
    mOperandCaches[operand].name = operand->GetName();
    mOperandCaches[operand].apiTensorRefCount = 0;
}

}  // namespace core
}  // namespace dtorch
