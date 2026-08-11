/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "dtorch/core/graph/logical_graph.h"
#include "operator.h"

namespace dtorch {
namespace core {

struct FuseParam : public OpParam {
    // TODO: logicalGraph support deep copy or not
    // LogicalGraph logicalGraph;
    OperatorType dominateOp;

public:
    FuseParam(OperatorType dominateOp = OperatorType::kActivation)
        : OpParam(OperatorType::kFuse), /*logicalGraph(),*/ dominateOp(dominateOp) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & dominateOp;
    }
};

class FuseOp : public Operator {
public:
public:
    FuseOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr), mInputNames(), mOutputNames() {}

    void InferOutputMetaInfo() const override;

private:
    mutable std::vector<std::string> mInputNames;
    mutable std::vector<std::string> mOutputNames;
};

}  // namespace core
}  // namespace dtorch
