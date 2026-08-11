/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "contiguous_op.h"

namespace dtorch {
namespace core {

void ContiguousOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 1);
    OperandY()->MetaDataSameAs(OperandX());
}

}  // namespace core
}  // namespace dtorch
