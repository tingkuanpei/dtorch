/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "placement_signature.h"

#include <functional>

#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"

namespace dtorch {
namespace core {

std::vector<std::vector<size_t>> PlacementSignature::Builder::GetAllCombination(
    const std::vector<size_t>& mCurrentOptionalIndexs) {
    DDebugAssert(mCurrentOptionalIndexs.size() > 0);
    std::vector<std::vector<size_t>> result;

    std::function<void(std::vector<size_t>, size_t)> GetAllCombinationImp;
    GetAllCombinationImp = [&](std::vector<size_t> prefix, size_t idx) {
        if (idx >= mCurrentOptionalIndexs.size()) {
            result.push_back(prefix);
            return;
        }

        GetAllCombinationImp(prefix, idx + 1);
        prefix.push_back(mCurrentOptionalIndexs[idx]);
        GetAllCombinationImp(prefix, idx + 1);
        return;
    };

    GetAllCombinationImp(std::vector<size_t>(), 0);
    return result;
}

void PlacementSignature::Builder::Build() {
    DAlwaysAssertMsg(mInputSize == mCurrentInputSignature.size(), "PlacementSignature's input must have same size");
    DAlwaysAssertMsg(mOutputSize == mCurrentOutputSignature.size(), "PlacementSignature's output must have same size");

    for (auto idx : mCurrentOptionalIndexs) {
        mAllOptionalIndexs.insert(idx);
    }

    if (mCurrentOptionalIndexs.size() > 0) {
        for (const auto& vec : GetAllCombination(mCurrentOptionalIndexs)) {
            InputPlacementSignature inputs = mCurrentInputSignature;
            for (auto it : vec) {
                DDebugAssert(inputs.size() > it);
                inputs[it] = Placement::Optional();
            }
            mPlacementSignaturePair.push_back(std::make_pair(inputs, mCurrentOutputSignature));
        }
    } else {
        mPlacementSignaturePair.push_back(std::make_pair(mCurrentInputSignature, mCurrentOutputSignature));
    }

    mCurrentOptionalIndexs.clear();
    mCurrentInputSignature.clear();
    mCurrentOutputSignature.clear();
}

PlacementSignature PlacementSignature::Builder::Finish() {
    DDebugAssertMsg(mCurrentInputSignature.empty() && mCurrentOutputSignature.empty(),
                    "Call Builder::Build() before Builder::Finish()");

    // Add Replicate
    for (size_t i = 0; i < mInputSize; i++) {
        if (mAllOptionalIndexs.count(i)) {
            this->AddOptionalInput(Replicate());
        } else {
            this->AddInput(Replicate());
        }
    }
    for (size_t i = 0; i < mOutputSize; i++) {
        this->AddOutput(Replicate());
    }
    this->Build();

    // Generate PlacementSignature
    PlacementSignature placementSignature(mInputSize, mOutputSize);
    for (auto it : mPlacementSignaturePair) {
        placementSignature.AddPlacementSignaturePair(it);
    }
    return placementSignature;
}

PlacementSignature::PlacementSignature(size_t inputSize, size_t outputSize)
    : mInputSize(inputSize), mOutputSize(outputSize), mAllSignature(), mSignatureMap() {}

bool PlacementSignature::Match(const std::vector<PlacementSeq>& inputs, std::vector<PlacementSeq>& outputs,
                               bool keepSubSplitCoordinates) const {
    DDebugAssert(inputs.size() == mInputSize);
    std::vector<std::vector<Placement>> result;

    size_t dims = inputs[0].Size();
    for (size_t i = 0; i < dims; i++) {
        InputPlacementSignature inputSignature = PlacementSignature::GetInputPlacementSignature(inputs, i);
        uint64_t key = GetHashKey(inputSignature);
        auto it = mSignatureMap.find(key);
        if (it == mSignatureMap.end()) {
            return false;
        }

        // TODO: Add support for placement with SubSplitCoordinates.
        // TEMP CODE, REMOVE THIS: if input is split with SubSplitCoordinates, add same SubSplitCoordinates for output
        // split
        DDebugAssert(it != mSignatureMap.end());
        OutputPlacementSignature copyOutputSignature = it->second;
        if (keepSubSplitCoordinates) {
            for (size_t i = 0; i < inputSignature.size(); i++) {
                if (inputSignature[i].HasSubSplitCoordinates()) {
                    int64_t subSplitCoordinates = inputSignature[i].GetSubSplitCoordinates();
                    for (size_t j = 0; j < copyOutputSignature.size(); j++) {
                        if (copyOutputSignature[j].IsShard()) {
                            copyOutputSignature[j].SetSubSplitCoordinates(subSplitCoordinates);
                        }
                    }
                    break;
                }
            }
        }

        SetOutputPlacementSignature(result, i, copyOutputSignature);
    }

    outputs.clear();
    for (size_t i = 0; i < result.size(); i++) {
        outputs.push_back(PlacementSeq(result[i]));
    }
    return true;
}

PlacementSignature::InputPlacementSignature PlacementSignature::GetInputPlacementSignature(
    const std::vector<PlacementSeq>& inputs, size_t dim) const {
    InputPlacementSignature inputPlacement;
    size_t numInput = inputs.size();

    for (size_t i = 0; i < numInput; i++) {
        DDebugAssert(inputs[i].Size() > dim);
        inputPlacement.push_back(inputs[i][dim]);
    }
    return inputPlacement;
}

void PlacementSignature::SetOutputPlacementSignature(std::vector<std::vector<Placement>>& outputs, size_t dim,
                                                     OutputPlacementSignature outputSignature) const {
    size_t outputSize = outputSignature.size();
    DDebugAssert(outputSize > 0);
    if (outputs.size() != outputSize) {
        outputs.resize(outputSize);
    }

    for (size_t i = 0; i < outputSize; i++) {
        DDebugAssert(outputs[i].size() == dim);
        outputs[i].push_back(outputSignature[i]);
    }
}

void PlacementSignature::AddPlacementSignaturePair(const IOPlacementSignaturePair& pair) {
    DDebugAssert(pair.first.size() == mInputSize);
    DDebugAssert(pair.second.size() == mOutputSize);
    uint64_t key = GetHashKey(pair.first);
    mSignatureMap[key] = pair.second;
    mAllSignature.push_back(pair);
}

uint64_t PlacementSignature::GetHashKey(const InputPlacementSignature& inputPlacement) const {
    DAlwaysAssertMsg(inputPlacement.size() < 8, "Input size too large: " + std::to_string(inputPlacement.size()));

    uint64_t result = 0;
    for (size_t i = 0; i < inputPlacement.size(); i++) {
        result = (result << 8) | (inputPlacement[i].ToHashKey() & 0xFF);
    }

    return result;
}

std::string PlacementSignature::ToString() const {
    std::stringstream ss;
    for (size_t i = 0; i < mAllSignature.size(); i++) {
        const InputPlacementSignature& inSignature = mAllSignature[i].first;
        const OutputPlacementSignature& outSignature = mAllSignature[i].second;

        ss << "    signature" << i << " , inputs: ";
        for (size_t inIdx = 0; inIdx < inSignature.size(); inIdx++) {
            ss << inSignature[inIdx] << ", ";
        }
        ss << "outputs: ";
        for (size_t outIdx = 0; outIdx < outSignature.size(); outIdx++) {
            ss << outSignature[outIdx] << ", ";
        }
        ss << std::endl;
    }
    return ss.str();
}

}  // namespace core
}  // namespace dtorch
