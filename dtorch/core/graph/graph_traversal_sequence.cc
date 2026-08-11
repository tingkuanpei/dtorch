/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "graph_traversal_sequence.h"

namespace dtorch {
namespace core {

GraphTraversalSequence::GraphTraversalSequence() : mTraversalSeq(), mNodeMap() {}

GraphTraversalSequence::GraphTraversalSequence(const std::vector<const Operator*>& sequences)
    : mTraversalSeq(), mNodeMap() {
    FromVec(sequences);
}

GraphTraversalSequence::~GraphTraversalSequence() = default;

void GraphTraversalSequence::FromVec(const std::vector<const Operator*>& sequences) {
    Clear();

    auto lastIter = mTraversalSeq.begin();
    for (const auto& op : sequences) {
        lastIter = this->InsertAfter(lastIter, op);
    }
}

std::vector<const Operator*> GraphTraversalSequence::ToVec() const {
    return std::vector<const Operator*>(mTraversalSeq.begin(), mTraversalSeq.end());
}

std::unordered_set<const Operator*> GraphTraversalSequence::ToSet() const {
    return std::unordered_set<const Operator*>(mTraversalSeq.begin(), mTraversalSeq.end());
}

void GraphTraversalSequence::Clear() noexcept {
    mTraversalSeq.clear();
    mNodeMap.clear();
}

GraphTraversalSequence::Iterator GraphTraversalSequence::GetIterator(const Operator* op) const noexcept {
    auto it = mNodeMap.find(op);
    DDebugAssert(it != mNodeMap.end());
    return it->second;
}

GraphTraversalSequence::ConstIterator GraphTraversalSequence::GetConstIterator(const Operator* op) const noexcept {
    auto it = mNodeMap.find(op);
    DDebugAssert(it != mNodeMap.end());
    DDebugAssert(*(it->second) == op);
    ConstIterator constIt = it->second;
    return constIt;
}

GraphTraversalSequence::Iterator GraphTraversalSequence::InsertAfter(ConstIterator pos, const Operator* op) {
    DDebugAssert(mNodeMap.count(op) == 0);
    Iterator nextElementIter;
    if (pos == mTraversalSeq.end()) {
        nextElementIter = mTraversalSeq.insert(pos, op);
    } else {
        nextElementIter = mTraversalSeq.insert(std::next(pos), op);
    }
    mNodeMap[op] = nextElementIter;
    return nextElementIter;
}

GraphTraversalSequence::Iterator GraphTraversalSequence::Erase(ConstIterator pos) {
    mNodeMap.erase(*pos);
    return mTraversalSeq.erase(pos);
}

}  // namespace core
}  // namespace dtorch
