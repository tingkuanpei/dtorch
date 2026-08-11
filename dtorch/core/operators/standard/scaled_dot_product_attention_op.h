/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

struct SdpaParam : public OpParam {
    bool isCausal;
    std::optional<double> scale;
    bool enableGqa;
    std::optional<SdpaOption> option;

public:
    SdpaParam()
        : OpParam(OperatorType::kSdpa), isCausal(false), scale(std::nullopt), enableGqa(false), option(std::nullopt) {}

    SdpaParam(bool isCausal, std::optional<double> scale, bool enableGqa, std::optional<SdpaOption> option)
        : OpParam(OperatorType::kSdpa), isCausal(isCausal), scale(scale), enableGqa(enableGqa), option(option) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & isCausal;
        ar & scale;
        ar & enableGqa;
        ar & option;
    }
};

class SdpaOp : public Operator {
public:
public:
    SdpaOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    bool IsRequireInputSameDataKind() const override { return false; }

    void InferOutputMetaInfo() const override;

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;

    PlacementSignature GetPlacementSignature() const override;

    OperatorCost GetOperatorCost() const override;
};

}  // namespace core
}  // namespace dtorch
