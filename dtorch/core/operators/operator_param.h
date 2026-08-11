/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <vector>

#include "dtorch/common/logging.h"
#include "dtorch/common/type_cast.h"
#include "dtorch/core/type.h"

namespace dtorch {
namespace core {

// clang-format off
#define DTORCH_FOREACH_OPERATOR_TYPE(Func)              \
    Func(Activation                     ,  0)           \
    Func(BatchNorm                      ,  1)           \
    Func(BroadcastBinary                ,  2)           \
    Func(Conv                           ,  3)           \
    Func(Convert                        ,  4)           \
    Func(Dropout                        ,  5)           \
    Func(Expand                         ,  6)           \
    Func(Flatten                        ,  7)           \
    Func(Fuse                           ,  8)           \
    Func(Linear                         ,  9)           \
    Func(Pooling                        , 10)           \
    Func(Reduce                         , 11)           \
    Func(Reshape                        , 12)           \
    Func(BaseMath                       , 13)           \
    Func(Squeeze                        , 14)           \
    Func(Unsqueeze                      , 15)           \
    Func(Embedding                      , 16)           \
    Func(Normalization                  , 17)           \
    Func(Concat                         , 18)           \
    Func(Transpose                      , 19)           \
    Func(View                           , 20)           \
    Func(Contiguous                     , 21)           \
    Func(GetItem                        , 22)           \
    Func(Sdpa                           , 23)           \
    Func(Create                         , 24)           \
    Func(Copy                           , 25)           \
    Func(Chunk                          , 26)           \
    Func(Einsum                         , 27)           \
    Func(Permute                        , 28)           \
    Func(Interpolate                    , 29)           \
    Func(Repeat                         , 30)           \
    Func(Pad                            , 31)           \
    Func(Clamp                          , 32)           \
    Func(Nvtx                           , 33)           \
    Func(Where                          , 34)           \
    Func(MaxMin                         , 35)           \
    Func(Matmul                         , 36)           \
    Func(Softmax                        , 37)           \
    Func(Outer                          , 38)           \
    Func(RepeatInterleave               , 39)           \
    Func(ApplyRotaryEmb                 , 40)           \
    Func(SiluLinearChunk                , 41)           \
    Func(LayerNormMulAdd                , 42)           \
    Func(SubGraph                       , 43)           \
    Func(Memory                         , 44)           \
    Func(SetItem                        , 45)           \
    Func(Clone                          , 46)           \
    Func(GetTensor                      , 47)           \
    Func(Masked                         , 48)           \
    Func(Sync                           , 49)

// clang-format on

enum class OperatorType {
#define DTORCH_FUNC(Name, Value) k##Name = Value,
    DTORCH_FOREACH_OPERATOR_TYPE(DTORCH_FUNC)
#undef DTORCH_FUNC

        kCount
};

const std::string& OpTypeToString(OperatorType opType);

OperatorType OpTypeFromString(const std::string& str);

std::ostream& operator<<(std::ostream& os, OperatorType opType);

// 需要根据各个厂商高性能计算库的实现，以及PyTorch、TensorFlo、ONNX的接口，合理定义各个operator
//
// All param in OpParam should be float or int32_t, and same as ONNX definition
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm
// float type should be enough.
//
// 使用类似boost中侵入式Serialization的方式保存模型
struct OpParam {
public:
    OpParam(OperatorType opType) : mOpType(opType) {}

    virtual ~OpParam() = default;

    OperatorType GetOpType() const noexcept { return mOpType; }

    void SetOpType(OperatorType operatorType) noexcept { mOpType = operatorType; }

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & mOpType;
    }

private:
    OperatorType mOpType;
};

template <OperatorType kOperatorType>
struct NoElementOpParam : public OpParam {
public:
    NoElementOpParam() : OpParam(kOperatorType) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
    }
};

class OpParamUtil {
public:
    static std::vector<int64_t> IntOrIntArrayTo2DParam(const IntOrIntArray& intArray, const std::string& transferType);

    static std::vector<int64_t> IntOrIntArrayTo2DPad(const IntOrIntArray& intArray);

    static bool IsPadEmptyOrZero(const std::vector<int64_t>& pads);
};

}  // namespace core
}  // namespace dtorch
