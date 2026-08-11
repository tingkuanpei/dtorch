/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <optional>

#include "../operator.h"

namespace dtorch {
namespace core {

struct PadParam : public OpParam {
    std::vector<int64_t> pad;
    std::string mode;
    std::optional<double> value;

public:
    PadParam() : OpParam(OperatorType::kPad), pad(), mode(), value() {}

    PadParam(const std::vector<int64_t>& pad, const std::string& mode, const std::optional<double>& value)
        : OpParam(OperatorType::kPad), pad(pad), mode(mode), value(value) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & pad;
        ar & mode;
        ar & value;
    }
};

class PadOp : public Operator {
public:
public:
    PadOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    void InferOutputMetaInfo() const override;

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;
};

}  // namespace core
}  // namespace dtorch
