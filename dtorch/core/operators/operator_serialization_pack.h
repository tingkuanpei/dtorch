/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "dtorch/core/operators/operator_param.h"
#include "dtorch/external/boost/boost_serialization.h"
#include "dtorch/external/boost/boost_serialization_torch.h"
#include "fuse_op.h"
#include "fused_compile/apply_rotary_emb_op.h"
#include "fused_compile/layer_norm_mul_add.h"
#include "fused_compile/silu_linear_chunk.h"
#include "standard/activation_op.h"
#include "standard/base_math_op.h"
#include "standard/batch_norm_op.h"
#include "standard/broadcast_binary_op.h"
#include "standard/chunk_op.h"
#include "standard/clamp_op.h"
#include "standard/clone_op.h"
#include "standard/concat_op.h"
#include "standard/contiguous_op.h"
#include "standard/conv_op.h"
#include "standard/convert_op.h"
#include "standard/copy_op.h"
#include "standard/create_op.h"
#include "standard/dropout_op.h"
#include "standard/einsum_op.h"
#include "standard/embedding_op.h"
#include "standard/expand_op.h"
#include "standard/flatten_op.h"
#include "standard/get_item_op.h"
#include "standard/interpolate_op.h"
#include "standard/linear_op.h"
#include "standard/masked_op.h"
#include "standard/matmul_op.h"
#include "standard/max_min_op.h"
#include "standard/normalization_op.h"
#include "standard/outer_op.h"
#include "standard/pad_op.h"
#include "standard/permute_op.h"
#include "standard/pooling_op.h"
#include "standard/reduce_op.h"
#include "standard/repeat_interleave_op.h"
#include "standard/repeat_op.h"
#include "standard/reshape_op.h"
#include "standard/scaled_dot_product_attention_op.h"
#include "standard/set_item_op.h"
#include "standard/softmax_op.h"
#include "standard/squeeze_op.h"
#include "standard/transpose_op.h"
#include "standard/unsqueeze_op.h"
#include "standard/view_op.h"
#include "standard/where_op.h"
#include "subgraph_op.h"
#include "system/get_tensor_op.h"
#include "system/memory_op.h"
#include "system/nvtx_op.h"
#include "system/sync_op.h"

namespace dtorch {
namespace core {

class OperatorSerializationPack {
public:
    std::string opName;
    uint64_t uniqueId;
    std::shared_ptr<OpParam> opParam;
    std::vector<uintptr_t> uintInputOperands;
    std::vector<uintptr_t> uintOutputOperands;

public:
    OperatorSerializationPack()
        : opName(""), uniqueId(OperatorIdManager::kNoValue), opParam(), uintInputOperands(), uintOutputOperands() {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/);

    std::string ToString() const;

    DTORCH_API_FORCEINLINE friend std::ostream& operator<<(std::ostream& os, const OperatorSerializationPack& pack) {
        os << pack.ToString();
        return os;
    }
};

template <class Archive>
void OperatorSerializationPack::serialize(Archive& ar, const unsigned int /*version*/) {
    ar & opName;
    ar & uniqueId;
    ar & uintInputOperands;
    ar & uintOutputOperands;

    OperatorType opType = OperatorType::kActivation;
    if (opParam) {
        opType = opParam->GetOpType();
    }
    ar & opType;

    if constexpr (Archive::is_saving::value) {
        // Example:
        // case OperatorType::kCreate: {
        //      CreateParam param = dynamic_cast<CreateParam&>(*opParam);
        //      ar & param;
        // } break;
        switch (opType) {
#define DTORCH_FUNC(Name, Value)                                               \
    case OperatorType::k##Name: {                                              \
        const Name##Param& param = dynamic_cast<const Name##Param&>(*opParam); \
        ar & param;                                                            \
    } break;
            DTORCH_FOREACH_OPERATOR_TYPE(DTORCH_FUNC)
#undef DTORCH_FUNC

            default:
                DLogError() << "Unsupport opType: " << EnumAsInteger(opType);
                DUnimplemented();
                break;
        }
    } else {
        // Example:
        // case OperatorType::kCreate: {
        //      CreateParam param(CreateKind::kArange, {}, DataKind::kFloat16, DeviceMesh(), PlacementSeq());
        //      ar & param;
        //      opParam = std::make_shared<CreateParam>(param);
        // } break;
        switch (opType) {
#define DTORCH_FUNC(Name, Value)                                   \
    case OperatorType::k##Name: {                                  \
        Name##Param param;                                         \
        ar & param;                                                \
        opParam = std::make_shared<Name##Param>(std::move(param)); \
    } break;
            DTORCH_FOREACH_OPERATOR_TYPE(DTORCH_FUNC)
#undef DTORCH_FUNC

            default:
                DLogError() << "Unsupport opType: " << EnumAsInteger(opType);
                DUnimplemented();
                break;
        }
    }
}

}  // namespace core
}  // namespace dtorch
