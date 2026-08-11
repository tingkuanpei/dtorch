/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <functional>
#include <memory>
#include <unordered_map>

#include "operator.h"

namespace dtorch {
namespace core {

class OperatorFactory {
public:
    DTORCH_API_FORCEINLINE static OperatorFactory& GetSingleton() {
        static OperatorFactory singleton;
        return singleton;
    }

public:
    std::unique_ptr<Operator> NewOperatorOrThrow(std::shared_ptr<OpParam> opParamPtr, const OperandArray& inputOperands,
                                                 std::optional<uint64_t> uniqueId = std::nullopt);

private:
    OperatorFactory();

    template <typename OperatorClass>
    void RegisterOpConstructor(OperatorType opType);

    std::unique_ptr<Operator> ConstructOperator(std::shared_ptr<OpParam> opParamPtr);

private:
    using OpConstructorFunc = std::function<std::unique_ptr<Operator>(std::shared_ptr<OpParam>)>;

    std::unordered_map<OperatorType, OpConstructorFunc> mOpConstructorMap;
};

}  // namespace core
}  // namespace dtorch
