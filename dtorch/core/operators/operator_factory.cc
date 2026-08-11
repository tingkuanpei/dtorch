/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "operator_factory.h"

#include <memory>
#include <sstream>

#include "dtorch/common/debug.h"
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
#include "system/get_tensor_op.h"
#include "system/memory_op.h"
#include "system/nvtx_op.h"
#include "system/sync_op.h"

namespace dtorch {
namespace core {

template <typename OperatorClass>
void OperatorFactory::RegisterOpConstructor(OperatorType opType) {
    auto opConstructorFunc = [](std::shared_ptr<OpParam> opParamPtr) -> std::unique_ptr<Operator> {
        return std::unique_ptr<Operator>(new OperatorClass(opParamPtr));
    };

    if (mOpConstructorMap.count(opType) > 0) {
        DLogFatal() << "Can't RegisterOpConstructor at same operator type: " << EnumAsInteger<OperatorType>(opType)
                    << "." << "Please check the OperatorType value.";
    }

    mOpConstructorMap[opType] = opConstructorFunc;
}

OperatorFactory::OperatorFactory() : mOpConstructorMap() {
    RegisterOpConstructor<ActivationOp>(OperatorType::kActivation);
    RegisterOpConstructor<BaseMathOp>(OperatorType::kBaseMath);
    RegisterOpConstructor<BatchNormOp>(OperatorType::kBatchNorm);
    RegisterOpConstructor<BroadcastBinaryOp>(OperatorType::kBroadcastBinary);
    RegisterOpConstructor<ChunkOp>(OperatorType::kChunk);
    RegisterOpConstructor<ClampOp>(OperatorType::kClamp);
    RegisterOpConstructor<CloneOp>(OperatorType::kClone);
    RegisterOpConstructor<ConvOp>(OperatorType::kConv);
    RegisterOpConstructor<CopyOp>(OperatorType::kCopy);
    RegisterOpConstructor<ConcatOp>(OperatorType::kConcat);
    RegisterOpConstructor<ContiguousOp>(OperatorType::kContiguous);
    RegisterOpConstructor<ConvertOp>(OperatorType::kConvert);
    RegisterOpConstructor<CreateOp>(OperatorType::kCreate);
    RegisterOpConstructor<DropoutOp>(OperatorType::kDropout);
    RegisterOpConstructor<EmbeddingOp>(OperatorType::kEmbedding);
    RegisterOpConstructor<EinsumOp>(OperatorType::kEinsum);
    RegisterOpConstructor<ExpandOp>(OperatorType::kExpand);
    RegisterOpConstructor<FlattenOp>(OperatorType::kFlatten);
    RegisterOpConstructor<GetItemOp>(OperatorType::kGetItem);
    RegisterOpConstructor<GetTensorOp>(OperatorType::kGetTensor);
    RegisterOpConstructor<SetItemOp>(OperatorType::kSetItem);
    RegisterOpConstructor<LinearOp>(OperatorType::kLinear);
    RegisterOpConstructor<TransposeOp>(OperatorType::kTranspose);
    RegisterOpConstructor<InterpolateOp>(OperatorType::kInterpolate);
    RegisterOpConstructor<NormalizationOp>(OperatorType::kNormalization);
    RegisterOpConstructor<NvtxOp>(OperatorType::kNvtx);
    RegisterOpConstructor<SyncOp>(OperatorType::kSync);
    RegisterOpConstructor<MatmulOp>(OperatorType::kMatmul);
    RegisterOpConstructor<MaxMinOp>(OperatorType::kMaxMin);
    RegisterOpConstructor<MemoryOp>(OperatorType::kMemory);
    RegisterOpConstructor<OuterOp>(OperatorType::kOuter);
    RegisterOpConstructor<PadOp>(OperatorType::kPad);
    RegisterOpConstructor<PermuteOp>(OperatorType::kPermute);
    RegisterOpConstructor<PoolingOp>(OperatorType::kPooling);
    RegisterOpConstructor<ReduceOp>(OperatorType::kReduce);
    RegisterOpConstructor<RepeatInterleaveOp>(OperatorType::kRepeatInterleave);
    RegisterOpConstructor<ReshapeOp>(OperatorType::kReshape);
    RegisterOpConstructor<SdpaOp>(OperatorType::kSdpa);
    RegisterOpConstructor<SoftmaxOp>(OperatorType::kSoftmax);
    RegisterOpConstructor<SqueezeOp>(OperatorType::kSqueeze);
    RegisterOpConstructor<RepeatOp>(OperatorType::kRepeat);
    RegisterOpConstructor<UnsqueezeOp>(OperatorType::kUnsqueeze);
    RegisterOpConstructor<ViewOp>(OperatorType::kView);
    RegisterOpConstructor<WhereOp>(OperatorType::kWhere);
    RegisterOpConstructor<MaskedOp>(OperatorType::kMasked);

    // Fused compile operator
    RegisterOpConstructor<ApplyRotaryEmbOp>(OperatorType::kApplyRotaryEmb);
    RegisterOpConstructor<SiluLinearChunkOp>(OperatorType::kSiluLinearChunk);
    RegisterOpConstructor<LayerNormMulAddOp>(OperatorType::kLayerNormMulAdd);
}

std::unique_ptr<Operator> OperatorFactory::ConstructOperator(std::shared_ptr<OpParam> opParamPtr) {
    OperatorType opType = opParamPtr->GetOpType();
    const auto& iterator = mOpConstructorMap.find(opType);

    if (iterator == mOpConstructorMap.end()) {
        std::stringstream ss;
        ss << "Unsupported operator: " << OpTypeToString(opType) << std::endl;
        ss << "Candidate: ";
        for (const auto& it : mOpConstructorMap) {
            ss << OpTypeToString(it.first) << ", ";
        }
        ss << std::endl;

        DLogFatal() << ss.str();
    }

    return iterator->second(std::move(opParamPtr));
}

std::unique_ptr<Operator> OperatorFactory::NewOperatorOrThrow(std::shared_ptr<OpParam> opParamPtr,
                                                              const OperandArray& inputOperands,
                                                              std::optional<uint64_t> uniqueId) {
    std::unique_ptr<core::Operator> op = ConstructOperator(std::move(opParamPtr));
    if (uniqueId.has_value()) {
        op->SetUniqueId(uniqueId.value());
    } else {
        op->AssignUniqueId();
    }
    op->SetInputOperands(inputOperands);
    op->CreateOutputOperands();

    try {
        op->Infer();
    } catch (std::exception& e) {
        std::stringstream ss;
        ss << "Add operator failed: " << e.what();
        throw std::invalid_argument(ss.str());
    }

    op->UpdateIOOperandTopology();

    return op;
}

}  // namespace core
}  // namespace dtorch
