/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <list>

#include "dtorch/core/operators/operator.h"

namespace dtorch {
namespace core {

// std::list with searching element in average constant-time complexity support.
class GraphTraversalSequence {
public:
    using Iterator = std::list<const Operator*>::iterator;
    using ConstIterator = std::list<const Operator*>::const_iterator;
    using ReverseIterator = std::list<const Operator*>::reverse_iterator;
    using ConstReverseIterator = std::list<const Operator*>::const_reverse_iterator;

public:
    GraphTraversalSequence();

    GraphTraversalSequence(const std::vector<const Operator*>& sequences);

    ~GraphTraversalSequence();

    DTORCH_DISABLE_COPY_AND_MOVE(GraphTraversalSequence);

    void FromVec(const std::vector<const Operator*>& sequences);

    std::vector<const Operator*> ToVec() const;

    std::unordered_set<const Operator*> ToSet() const;

    void Clear() noexcept;

    DTORCH_FORCEINLINE size_t Size() const noexcept { return mTraversalSeq.size(); }

    DTORCH_FORCEINLINE bool Empty() const noexcept { return mTraversalSeq.empty(); }

    DTORCH_FORCEINLINE bool CountOperator(const Operator* op) const noexcept { return mNodeMap.count(op) > 0; }

    Iterator GetIterator(const Operator* op) const noexcept;
    ConstIterator GetConstIterator(const Operator* op) const noexcept;

    Iterator InsertAfter(ConstIterator pos, const Operator* op);

    Iterator Erase(ConstIterator pos);

    DTORCH_FORCEINLINE Iterator begin() noexcept { return mTraversalSeq.begin(); }
    DTORCH_FORCEINLINE ConstIterator begin() const noexcept { return mTraversalSeq.begin(); }
    DTORCH_FORCEINLINE ConstIterator cbegin() const noexcept { return mTraversalSeq.cbegin(); }

    DTORCH_FORCEINLINE ReverseIterator rbegin() noexcept { return mTraversalSeq.rbegin(); }
    DTORCH_FORCEINLINE ConstReverseIterator rbegin() const noexcept { return mTraversalSeq.rbegin(); }
    DTORCH_FORCEINLINE ConstReverseIterator crbegin() const noexcept { return mTraversalSeq.crbegin(); }

    DTORCH_FORCEINLINE Iterator end() noexcept { return mTraversalSeq.end(); }
    DTORCH_FORCEINLINE ConstIterator end() const noexcept { return mTraversalSeq.end(); }
    DTORCH_FORCEINLINE ConstIterator cend() const noexcept { return mTraversalSeq.cend(); }

    DTORCH_FORCEINLINE ReverseIterator rend() noexcept { return mTraversalSeq.rend(); }
    DTORCH_FORCEINLINE ConstReverseIterator rend() const noexcept { return mTraversalSeq.rend(); }
    DTORCH_FORCEINLINE ConstReverseIterator crend() const noexcept { return mTraversalSeq.crend(); }

private:
    std::list<const Operator*> mTraversalSeq;
    std::unordered_map<const Operator*, Iterator> mNodeMap;
};

using TraversalIter = GraphTraversalSequence::Iterator;
using ConstTraversalIter = GraphTraversalSequence::ConstIterator;
using ConstTraversalRange = std::pair<ConstTraversalIter, ConstTraversalIter>;

}  // namespace core
}  // namespace dtorch
