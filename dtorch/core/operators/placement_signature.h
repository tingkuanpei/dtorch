/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dtorch/common/debug.h"
#include "dtorch/common/utilities.h"
#include "dtorch/core/type.h"

namespace dtorch {
namespace core {

class PlacementSignature {
public:
    using InputPlacementSignature = std::vector<Placement>;
    using OutputPlacementSignature = std::vector<Placement>;
    using IOPlacementSignaturePair = std::pair<InputPlacementSignature, OutputPlacementSignature>;

    class Builder {
    public:
        Builder(size_t inputSize, size_t outputSize)
            : mInputSize(inputSize),
              mOutputSize(outputSize),
              mAllOptionalIndexs(),
              mPlacementSignaturePair(),
              mCurrentOptionalIndexs(),
              mCurrentInputSignature(),
              mCurrentOutputSignature() {}

        DTORCH_FORCEINLINE Builder& AddInput(Placement placement) {
            mCurrentInputSignature.push_back(placement);
            return *this;
        }

        DTORCH_FORCEINLINE Builder& AddInput(const std::string& str) { return AddInput(Placement(str)); }

        DTORCH_FORCEINLINE Builder& AddOptionalInput(Placement placement) {
            mCurrentOptionalIndexs.push_back(mCurrentInputSignature.size());
            mCurrentInputSignature.push_back(placement);
            return *this;
        }

        DTORCH_FORCEINLINE Builder& AddOptionalInput(const std::string& str) {
            return AddOptionalInput(Placement(str));
        }

        DTORCH_FORCEINLINE Builder& AddOutput(Placement placement) {
            mCurrentOutputSignature.push_back(placement);
            return *this;
        }

        DTORCH_FORCEINLINE Builder& AddOutput(const std::string& str) { return AddOutput(Placement(str)); }

        void Build();

        PlacementSignature Finish();

    private:
        std::vector<std::vector<size_t>> GetAllCombination(const std::vector<size_t>& mCurrentOptionalIndexs);

    private:
        const size_t mInputSize;
        const size_t mOutputSize;
        std::unordered_set<size_t> mAllOptionalIndexs;
        std::vector<IOPlacementSignaturePair> mPlacementSignaturePair;
        std::vector<size_t> mCurrentOptionalIndexs;
        InputPlacementSignature mCurrentInputSignature;
        OutputPlacementSignature mCurrentOutputSignature;
    };

public:
    PlacementSignature(size_t inputSize, size_t outputSize);

    DTORCH_DISABLE_COPY_AND_DEFAULT_MOVE(PlacementSignature);

    bool Match(const std::vector<PlacementSeq>& inputs, std::vector<PlacementSeq>& outputs,
               bool keepSubSplitCoordinates) const;

    std::string ToString() const;

private:
    InputPlacementSignature GetInputPlacementSignature(const std::vector<PlacementSeq>& inputs, size_t dim) const;

    void SetOutputPlacementSignature(std::vector<std::vector<Placement>>& outputs, size_t dim,
                                     OutputPlacementSignature outputSignature) const;

    void AddPlacementSignaturePair(const IOPlacementSignaturePair& pair);

    uint64_t GetHashKey(const InputPlacementSignature& inputPlacement) const;

private:
    size_t mInputSize;
    size_t mOutputSize;
    std::vector<IOPlacementSignaturePair> mAllSignature;
    std::unordered_map<uint64_t, OutputPlacementSignature> mSignatureMap;
};

}  // namespace core
}  // namespace dtorch
