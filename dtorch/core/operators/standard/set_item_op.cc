/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "set_item_op.h"

#include <sstream>
#include <stdexcept>

#include "get_item_op.h"

namespace dtorch {
namespace core {

void SetItemOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 2);
    DDebugAssert(GetOutputSize() == 1);

    // 1. Check Shape
    // Validate the value tensor (input 1) is broadcastable to the indexed region.
    auto indexVec = GetOpParam<SetItemParam>().indexVec;
    const Shape& inShape = OperandA()->GetShape();
    Shape indexedShape = GetItemOp::ComputeIndexedShape(inShape, indexVec);
    const Shape& valueShape = OperandB()->GetShape();
    Shape broadcastShape = Shape::BroadcastOutputShape(indexedShape, valueShape);
    if (broadcastShape != indexedShape) {
        std::stringstream ss;
        ss << "SetItemOp value shape " << valueShape.ToString() << " is not broadcastable to indexed shape "
           << indexedShape.ToString();
        throw std::invalid_argument(ss.str());
    }

    // 2. Set OperandY
    OperandY()->MetaDataSameAs(OperandA());

    // 3. Set mIndexVec
    mIndexVec = std::move(indexVec);
}

}  // namespace core
}  // namespace dtorch
