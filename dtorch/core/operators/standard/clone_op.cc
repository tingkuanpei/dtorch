/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "clone_op.h"

namespace dtorch {
namespace core {

void CloneOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 1);
    OperandY()->MetaDataSameAs(OperandX());
}

}  // namespace core
}  // namespace dtorch
