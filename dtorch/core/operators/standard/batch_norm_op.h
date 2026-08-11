/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

struct BatchNormParam : public OpParam {
    double epsilon;
    double momentum;
    OperatorFormat format;

public:
    BatchNormParam(double epsilon = 1e-5, double momentum = 0.1,
                   OperatorFormat format = api::cpp::OperatorFormat::kNCHW)
        : OpParam(OperatorType::kBatchNorm), epsilon(epsilon), momentum(momentum), format(format) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & epsilon;
        ar & momentum;
        ar & format;
    }
};

class BatchNormOp : public Operator {
public:
public:
    BatchNormOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    void InferOutputMetaInfo() const override;

    std::string GetDescribeString() const override;

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;
};

}  // namespace core
}  // namespace dtorch
