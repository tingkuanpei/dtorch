/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include <fstream>
#include <memory>

#include "dtorch/api/cpp/distributed_spec.h"
#include "dtorch/api/cpp/shape.h"
#include "dtorch/api/cpp/stride.h"
#include "dtorch/common/logging.h"
#include "dtorch/core/operators/operator.h"
#include "dtorch/core/operators/operator_serialization_pack.h"
#include "dtorch/external/boost/boost_serialization.h"
#include "dtorch/external/boost/boost_serialization_torch.h"
#include "test.h"

using namespace dtorch::api::cpp;
using namespace dtorch::core;
using namespace dtorch::external::boost;

TEST(SerializationTest, SimpleTest) {
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    std::optional<double> optionalValue(10.0);
    const Shape shape({1, 2, 3});
    const Stride stride({1, 2, 3});
    const DeviceMesh deviceMesh(DeviceKind::kCpu, {0, 1, 2, 3});
    const PlacementSeq placementSeq({Placement("R")});
    const Generator generator(Device(DeviceKind::kGpu, 0));
    const CreateParam createParam(CreateKind::kRand, shape, DataKind::kFloat32, deviceMesh, placementSeq, generator);
    CreateOp op(std::make_unique<CreateParam>(createParam));
    op.CreateOutputOperands();
    op.AssignUniqueId();
    const OperatorSerializationPack opPack = op.GetOperatorSerializationPack();
    torch::Tensor torchTensor = torch::rand({4, 3});
    torch::Tensor emptyTensor = torch::rand({4, 0});
    {
        BinaryOArchive boa(ss);
        boa << optionalValue;
        boa << shape;
        boa << stride;
        boa << deviceMesh;
        boa << placementSeq;
        boa << createParam;
        boa << opPack;
        boa << torchTensor;
        boa << emptyTensor;
    }

    std::optional<double> newOptionalValue;
    Shape newShape;
    Stride newStride;
    DeviceMesh newDeviceMesh;
    PlacementSeq newPlacementSeq;
    CreateParam newCreateParam;
    OperatorSerializationPack newOpPack;
    torch::Tensor newTensor;
    torch::Tensor newEmptyTensor;
    {
        BinaryIArchive bia(ss);
        bia >> newOptionalValue;
        bia >> newShape;
        bia >> newStride;
        bia >> newDeviceMesh;
        bia >> newPlacementSeq;
        bia >> newCreateParam;
        bia >> newOpPack;
        bia >> newTensor;
        bia >> newEmptyTensor;
    }

    EXPECT_TRUE(optionalValue.has_value() == newOptionalValue.has_value());
    EXPECT_TRUE(optionalValue.value() == newOptionalValue.value());
    EXPECT_TRUE(shape == newShape);
    EXPECT_TRUE(stride == newStride);
    EXPECT_TRUE(deviceMesh == newDeviceMesh);
    EXPECT_TRUE(placementSeq == newPlacementSeq);

    // Check CreateParam
    EXPECT_TRUE(createParam.GetOpType() == newCreateParam.GetOpType());
    EXPECT_TRUE(createParam.createKind == newCreateParam.createKind);
    EXPECT_TRUE(createParam.shape == newCreateParam.shape);
    EXPECT_TRUE(createParam.dataKind == newCreateParam.dataKind);
    EXPECT_TRUE(createParam.deviceMesh == newCreateParam.deviceMesh);
    EXPECT_TRUE(createParam.placementSeq == newCreateParam.placementSeq);
    DDebugAssert(createParam.generator.has_value());
    DDebugAssert(newCreateParam.generator.has_value());
    EXPECT_TRUE(createParam.generator.value().GetDevice() == newCreateParam.generator.value().GetDevice());
    EXPECT_TRUE(torch::allclose(createParam.generator.value().GetTorchGenerator().get_state(),
                                newCreateParam.generator.value().GetTorchGenerator().get_state()));

    // Check OperatorSerializationPack
    EXPECT_TRUE(opPack.opName == newOpPack.opName);
    EXPECT_TRUE(opPack.uniqueId == newOpPack.uniqueId);
    EXPECT_TRUE(opPack.opParam->GetOpType() == newOpPack.opParam->GetOpType());
    EXPECT_TRUE(dynamic_cast<CreateParam&>(*opPack.opParam).shape ==
                dynamic_cast<CreateParam&>(*newOpPack.opParam).shape);
    EXPECT_TRUE(opPack.uintInputOperands == newOpPack.uintInputOperands);
    EXPECT_TRUE(opPack.uintOutputOperands == newOpPack.uintOutputOperands);

    // Check torch::Tensor
    EXPECT_TRUE(torch::allclose(torchTensor, newTensor));
    EXPECT_TRUE(torch::allclose(emptyTensor, newEmptyTensor));
}
