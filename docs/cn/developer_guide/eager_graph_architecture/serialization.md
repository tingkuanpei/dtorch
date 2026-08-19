# Operator 序列化与反序列化

DTorch 序列化机制服务于两个核心场景：

1. **跨进程算子调用**：Controller 进程将 Operator 序列化后通过 ZMQ PUB-SUB 发送给 Worker 进程执行
2. **模型保存与加载**：将计算图持久化到磁盘，后续恢复执行

DTorch 中所有 Operand（张量元信息节点）均由 Operator（计算节点）生成，因此序列化所有 Operator 即可完整表述整个计算图。

## 1. Boost.Serialization 序列化基础设施

DTorch 基于 **Boost.Serialization** 实现序列化，为 `OperatorParam`、`Shape`、`DeviceMesh`、`PlacementSeq`、`torch::Tensor` 等大量类实现了侵入式 `serialize()` 方法。这些类通过嵌套序列化自然地组合成完整的数据结构。

### 1.1 Boost.Serialization 封装

**源文件**: `dtorch/external/boost/boost_serialization.h`, `dtorch/api/cpp/serialization.h`

```cpp
// dtorch/external/boost/boost_serialization.h
// 引入 Boost.Serialization 核心头文件，提供归档类型别名
using BinaryOArchive = ::boost::archive::binary_oarchive;  // 序列化为二进制
using BinaryIArchive = ::boost::archive::binary_iarchive;  // 从二进制反序列化

// dtorch/api/cpp/serialization.h
// 所有需要使用序列化的类通过 friend Serialization 授权访问
using Serialization = boost::serialization::access;
```

DTorch 使用**侵入式序列化**模式：类声明 `friend Serialization` 并定义模板方法 `serialize(Archive& ar, ...)`，由 Boost 框架通过 `ar & member` 语法递归序列化每个成员。`ar &` 运算符在 `Archive::is_saving` 时为写入，`Archive::is_loading` 时为读取。

此外，`dtorch/external/boost/boost_serialization.h` 还为 `std::optional<T>` 提供了特化的序列化方法，使其也能通过 `ar & opt` 语法使用。

### 1.2 基础类型的序列化

#### Shape

**源文件**: `dtorch/api/cpp/shape.h`

```cpp
class Shape {
    // ...
    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & mShape;  // std::vector<DataType>，Boost 原生支持 vector 序列化
    }
private:
    std::vector<DataType> mShape;
};
```

#### Device / DeviceKey

**源文件**: `dtorch/api/cpp/device.h`

```cpp
struct Device {
    DeviceKind deviceKind;  // 枚举类型
    int64_t deviceId;

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & deviceKind;
        ar & deviceId;
        // 注：std::string deviceName 不参与序列化（运行时恢复）
    }
};

struct DeviceKey {  // POD 类型，用于高效哈希查找
    DeviceKind deviceKind;
    int64_t deviceId;

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        if constexpr (Archive::is_loading::value) {
            std::memset(this, 0, sizeof(DeviceKey));  // 加载前零初始化（内存对齐）
        }
        ar & deviceKind;
        ar & deviceId;
    }
};
```

#### SimpleArray

**源文件**: `dtorch/api/cpp/simple_array.h`

`SimpleArray` 是多维数组的基础数据结构，`DeviceMesh` 内部使用 `SimpleArray` 存储 GPU 拓扑：

```cpp
class SimpleArray {
    Shape mShape;
    std::unordered_map<std::string, size_t> mDimensionNamesMap;
    std::vector<int64_t> mData;
    std::unordered_set<int64_t> mDataSet;

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & mShape;
        ar & mDimensionNamesMap;
        ar & mData;
        ar & mDataSet;
    }
};
```

### 1.3 分布式规格类型的序列化

#### DeviceMesh

**源文件**: `dtorch/api/cpp/distributed_spec.h`

```cpp
class DeviceMesh {
    DeviceKind mDeviceKind;                    // CPU / GPU
    std::shared_ptr<const SimpleArray> mMesh;  // N 维设备数组，如 [2, 4] 表示 2×4 GPU 网格

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & mDeviceKind;
        ar & mMesh;  // 通过 shared_ptr 序列化 SimpleArray
    }
};
```

#### Placement

`Placement` 描述张量在 DeviceMesh 各维度上的分布方式：

```cpp
class Placement {
    bool mReplicate;          // 是否复制
    bool mPartial;            // 是否部分聚合
    bool mShard;              // 是否切分
    int64_t mShardIndex;      // 切分维度
    int64_t mSubSplitCoordinates;  // 子切分坐标

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & mReplicate;
        ar & mPartial;
        ar & mShard;
        ar & mShardIndex;
        ar & mSubSplitCoordinates;
    }
};
```

#### PlacementSeq

`PlacementSeq` 是一组 Placement 的序列，描述张量在 N 维 DeviceMesh 上的完整分布策略：

```cpp
class PlacementSeq {
    std::shared_ptr<std::vector<Placement>> mData;  // shared_ptr 避免深拷贝

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & mData;  // Boost 原生支持 shared_ptr<vector<T>>
    }
};
```

### 1.4 Torch 类型的序列化

**源文件**: `dtorch/external/boost/boost_serialization_torch.h`

#### torch::Tensor

```cpp
template <typename Archive>
void serialize(Archive& ar, ::torch::Tensor& tensor, const unsigned int /*version*/) {
    bool defined = tensor.defined();
    ar & defined;

    if (defined) {
        Shape shape;
        Device device;
        DataKind dataKind;
        std::vector<char> dataBuffer;

        if constexpr (Archive::is_saving::value) {
            shape = TorchUtil::GetShape(tensor);
            device = TorchUtil::GetDevice(tensor);
            dataKind = TorchUtil::GetDataKind(tensor);
            dataBuffer = TorchUtil::ToCharVec(tensor);   // 张量数据 → 字节数组
        }
        ar & shape;
        ar & device;
        ar & dataKind;
        ar & dataBuffer;

        if constexpr (Archive::is_loading::value) {
            tensor = TorchUtil::CreateTensor(shape, device, dataKind, dataBuffer);
        }
    }
}
```

`torch::Tensor` 的序列化将张量拆解为四元组 `(Shape, Device, DataKind, raw bytes)`，跨进程恢复时据此重建。

#### torch::Generator

随机数生成器的序列化保存其设备与状态张量：

```cpp
template <typename Archive>
void serialize(Archive& ar, ::torch::Generator& generator, const unsigned int /*version*/) {
    bool defined = generator.defined();
    ar & defined;
    if (defined) {
        Device device;
        ::torch::Tensor state;
        if constexpr (Archive::is_saving::value) {
            device = TorchUtil::ToDevice(generator.device());
            state = generator.get_state();
        }
        ar & device;
        ar & state;
        if constexpr (Archive::is_loading::value) {
            generator = *TorchUtil::GetGenerator(device);
            generator.set_state(state);
        }
    }
}
```

### 1.5 Scalar 的序列化

**源文件**: `dtorch/api/cpp/scalar.h`

`Scalar` 使用 union 存储多种数值类型，序列化时需要根据 `mActiveTag` 分派：

```cpp
class Scalar {
    union Value { int64_t s; uint64_t u; double d; } mValue;
    enum { HAS_S, HAS_U, HAS_D, HAS_NONE } mActiveTag;

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & mActiveTag;
        switch (mActiveTag) {
            case HAS_S: ar & mValue.s; break;
            case HAS_U: ar & mValue.u; break;
            case HAS_D: ar & mValue.d; break;
            default: break;
        }
    }
};
```

### 1.6 OpParam 派生类的序列化

#### OpParam 基类

**源文件**: `dtorch/core/operators/operator_param.h`

```cpp
struct OpParam {
    OpParam(OperatorType opType) : mOpType(opType) {}
    OperatorType GetOpType() const noexcept { return mOpType; }

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & mOpType;  // 仅序列化算子类型枚举
    }

private:
    OperatorType mOpType;
};
```

#### NoElementOpParam — 无额外参数的算子

部分算子（如 `Linear`、`Flatten`、`View`、`Permute` 等）仅需 OpType 即足以唯一确定行为，使用 `NoElementOpParam` 模板：

```cpp
template <OperatorType kOperatorType>
struct NoElementOpParam : public OpParam {
    NoElementOpParam() : OpParam(kOperatorType) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);  // 序列化基类（即 mOpType）
    }
};

using LinearParam = NoElementOpParam<OperatorType::kLinear>;
```

#### ConvParam — 带丰富参数的算子（示例）

**源文件**: `dtorch/core/operators/standard/conv_op.h`

```cpp
struct ConvParam : public OpParam {
    std::vector<int64_t> dilations;
    int64_t group;
    std::vector<int64_t> kernelSize;
    PaddingType paddingType;         // 枚举
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    OperatorFormat format;           // 枚举

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);  // 先序列化基类 mOpType
        ar & dilations;
        ar & group;
        ar & kernelSize;
        ar & paddingType;
        ar & pads;
        ar & strides;
        ar & format;
    }
};
```

**序列化嵌套关系**：每个 `OpParam` 派生类通过 `BaseObject<OpParam>(*this)` 先序列化基类的 `mOpType`，再序列化自身字段。自身字段中如 `pads`（`IntOrIntArray`，即 `std::vector<int64_t>`）由 Boost 原生支持；`paddingType`、`format` 等枚举由 Boost 自动转为整数序列化；`Shape`、`DeviceMesh` 等复合类型则递归调用其自身的 `serialize()` 方法。

#### CreateParam — 广泛的类型覆盖（示例）

**源文件**: `dtorch/core/operators/standard/create_op.h`

`CreateParam` 是覆盖类型最广的参数类，包含了 `Shape`、`DataKind`、`DeviceMesh`、`PlacementSeq`、`std::optional<Generator>`、`std::optional<torch::Tensor>` 等多种类型，展示了 DTorch 序列化体系的组合能力：

```cpp
struct CreateParam : public OpParam {
    CreateKind createKind;                    // 枚举
    Shape shape;                              // → Shape::serialize()
    DataKind dataKind;                        // 枚举
    DeviceMesh deviceMesh;                    // → DeviceMesh::serialize()
    PlacementSeq placementSeq;               // → PlacementSeq::serialize()
    std::optional<Generator> generator;       // → Generator::serialize()
    double doubleArg0, doubleArg1, doubleArg2;
    std::optional<torch::Tensor> torchValue;  // → torch::Tensor::serialize()

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & createKind;
        ar & shape;
        ar & dataKind;
        ar & deviceMesh;
        ar & placementSeq;
        ar & generator;
        ar & doubleArg0;
        // ...
        ar & torchValue;
    }
};
```

### 1.7 序列化类型总览

DTorch 中实现 `serialize()` 方法的类型覆盖了计算图描述的各个层面：

```
┌─────────────────────────────────────────────────────────────────┐
│  DTorch 序列化类型体系                                            │
│                                                                 │
│  基础类型                 分布式规格               Torch 桥接    │
│  ────────                ──────────              ──────────     │
│  Shape                   DeviceMesh              torch::Tensor  │
│  Device / DeviceKey      Placement               torch::Generator│
│  SimpleArray             PlacementSeq                           │
│  Scalar                                                        │
│  DataKind (enum)                                               │
│  IntOrIntArray (vector<int64_t>)                               │
│                                                                 │
│  Operator 参数体系                                               │
│  ────────────────                                               │
│  OpParam (基类)                                                 │
│    ├─ NoElementOpParam<T>  (Linear, Flatten, View, Permute...)  │
│    ├─ ConvParam            (dilations, group, kernel, pads...)  │
│    ├─ CreateParam          (shape, deviceMesh, generator...)    │
│    ├─ ActivationParam      (activationKind)                     │
│    ├─ BroadcastBinaryParam (binaryKind)                         │
│    ├─ ReduceParam          (dims, keepdim)                      │
│    ├─ MatmulParam          (transA, transB)                     │
│    ├─ ... (共 47 种 Operator 类型)                              │
│                                                                 │
│  OperatorSerializationPack  (拓扑信息 + OpParam)                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. OperatorSerializationPack — 序列化数据包

**源文件**: `dtorch/core/operators/operator_serialization_pack.h`

`OperatorSerializationPack` 是单个 Operator 的序列化载体，将 Operator 的拓扑信息和参数信息打包为可序列化的数据结构。

### 2.1 成员变量

```cpp
class OperatorSerializationPack {
public:
    std::string opName;                          // Operator 名称
    uint64_t uniqueId;                           // 全局唯一 ID
    std::shared_ptr<OpParam> opParam;            // Operator 参数（多态）
    std::vector<uintptr_t> uintInputOperands;    // 输入 Operand 指针（转为 uintptr_t）
    std::vector<uintptr_t> uintOutputOperands;   // 输出 Operand 指针（转为 uintptr_t）
};
```

| 成员 | 作用 |
|---|---|
| `opName` | Operator 的字符串标识，如 `"linear_0"`、`"relu_1"` |
| `uniqueId` | 由 `OperatorIdManager` 分配的全局唯一 ID，用于反序列化时重建 Operator |
| `opParam` | 指向 `OpParam` 基类的 `shared_ptr`，实际存储派生类如 `LinearParam`、`ConvParam` 等 |
| `uintInputOperands` | 输入 Operand 的裸指针转为 `uintptr_t`，用于跨进程重建拓扑关系 |
| `uintOutputOperands` | 输出 Operand 的裸指针转为 `uintptr_t`，反序列化时注册到 `mOperandMap` 中 |

### 2.2 从 Operator 构建 Pack

`Operator::GetOperatorSerializationPack()` 将 Operator 的运行时状态打包：

```cpp
// dtorch/core/operators/operator.cc:298
OperatorSerializationPack Operator::GetOperatorSerializationPack() {
    OperatorSerializationPack pack;
    pack.opName = mOpName;
    pack.uniqueId = GetUniqueId();
    pack.opParam = mOpParam;
    for (auto operand : mInputOperands) {
        pack.uintInputOperands.push_back(reinterpret_cast<uintptr_t>(operand.get()));
    }
    for (auto operand : mOutputOperands) {
        pack.uintOutputOperands.push_back(reinterpret_cast<uintptr_t>(operand.get()));
    }
    return pack;
}
```

**关键设计**：Operand 指针被 `reinterpret_cast` 为 `uintptr_t`，这使得跨进程时可以唯一标识 Operand。反序列化端通过 `mOperandMap`（`unordered_map<uintptr_t, shared_ptr<Operand>>`）维护指针到 Operand 对象的映射，从而重建计算图的拓扑连接。

### 2.3 OperatorSerializationPack 的序列化实现

`OperatorSerializationPack::serialize()` 的核心挑战是 **OpParam 的多态序列化**：`opParam` 是 `shared_ptr<OpParam>`，实际指向 47 种派生类之一。序列化时不能直接序列化基类指针——必须根据具体类型分派：

```cpp
template <class Archive>
void OperatorSerializationPack::serialize(Archive& ar, const unsigned int /*version*/) {
    ar & opName;
    ar & uniqueId;
    ar & uintInputOperands;
    ar & uintOutputOperands;

    // 先读写 opType 以确定派生类型
    OperatorType opType = OperatorType::kActivation;
    if (opParam) { opType = opParam->GetOpType(); }
    ar & opType;

    if constexpr (Archive::is_saving::value) {
        // 序列化：根据 opType dynamic_cast 到具体 *Param 类型后序列化
        switch (opType) {
#define DTORCH_FUNC(Name, Value)                                  \
    case OperatorType::k##Name: {                                 \
        Name##Param param = dynamic_cast<Name##Param&>(*opParam); \
        ar & param;                                               \
    } break;
            DTORCH_FOREACH_OPERATOR_TYPE(DTORCH_FUNC)
#undef DTORCH_FUNC
        }
    } else {
        // 反序列化：根据 opType 默认构造 *Param，反序列化填充，包装为 shared_ptr
        switch (opType) {
#define DTORCH_FUNC(Name, Value)                        \
    case OperatorType::k##Name: {                       \
        Name##Param param;                              \
        ar & param;                                     \
        opParam = std::make_shared<Name##Param>(param); \
    } break;
            DTORCH_FOREACH_OPERATOR_TYPE(DTORCH_FUNC)
#undef DTORCH_FUNC
        }
    }
}
```

**设计要点**：

- 使用 `if constexpr (Archive::is_saving::value)` 在编译期分支，序列化和反序列化走不同的代码路径
- `DTORCH_FOREACH_OPERATOR_TYPE` 宏展开为全部 47 种 Operator 类型的 case 分支
- 序列化时：`dynamic_cast` 到具体 `*Param` 类型后序列化——这会递归触发 1.6 节中各 `*Param` 的 `serialize()` 方法，进而触发 `BaseObject<OpParam>` → `OpParam::serialize()`，最终将所有字段写入归档
- 反序列化时：先默认构造空的 `*Param`，反序列化填充字段，再包装为 `shared_ptr<OpParam>`——其 `mOpType` 已在 `OpParam::serialize()` 中被反序列化恢复

## 3. 序列化流程

### 3.1 发送端：RemoteRunnerPublisher::Execute

**源文件**: `dtorch/external/zmq/remote_runner_publisher.cc:37`

Controller 通过 `RemoteRunnerPublisher` 将 Operator 序列化后经 ZMQ PUB socket 广播给所有 Worker：

```cpp
void RemoteRunnerPublisher::Execute(
    const std::vector<std::shared_ptr<core::Operator>>& ops,
    const std::vector<const core::Operand*>& noHoldOperands) {

    // Step 1: 将每个 Operator 转换为 OperatorSerializationPack
    std::vector<core::OperatorSerializationPack> opPacks;
    std::vector<uintptr_t> noHoldOperandPtrs;
    for (const auto& op : ops) {
        opPacks.push_back(op->GetOperatorSerializationPack());
    }
    for (const auto& operand : noHoldOperands) {
        noHoldOperandPtrs.push_back(reinterpret_cast<uintptr_t>(operand));
    }

    // Step 2: 使用 Boost BinaryOArchive 序列化为二进制
    std::stringstream ss(std::ios::out | std::ios::binary);
    boost::BinaryOArchive boa(ss);
    boa << opPacks;
    boa << noHoldOperandPtrs;
    std::string serializedData = ss.str();

    // Step 3: 通过 ZMQ PUB 多帧消息发送
    int64_t messageId = PublishMessageIdManager::GetSingleton().GetIdAndIncrement(mImplPtr->address);
    const std::array<::zmq::const_buffer, 3> send_msgs = {
        ::zmq::buffer(std::to_string(messageId)),          // Frame 0: 消息 ID
        ::zmq::buffer(RemoteRunnerPublisher::kExecuteStr), // Frame 1: 消息类型 "publisherExecute"
        ::zmq::buffer(serializedData.data(), serializedData.size())  // Frame 2: 序列化数据
    };
    SendMultipart(mImplPtr->publisher, send_msgs);
}
```

**流程图**：

```
Operator[]  ──→  OperatorSerializationPack[]  ──→  BinaryOArchive  ──→  ZMQ PUB (3-frame multipart)
                      ↑                                        │
            GetOperatorSerializationPack()              boost::binary_oarchive
                                                        (std::stringstream binary)
```

**`noHoldOperands` 的作用**：某些 Operand 仅用于临时传递（如返回给 Client 的结果张量），Worker 执行完对应的读取操作后即可释放。这些 Operand 的指针也被序列化发送，Worker 端在 `ExecuteSerialization` 执行后将其从 `mOperandMap` 中移除。

### 3.2 接收端：RemoteRunner

**源文件**: `dtorch/core/runner/remote/remote_runner.cc:67`

Worker 端的 `RemoteRunner` 在 `AsyncMain` 循环中通过 `RemoteRunnerSubscriber`（SUB socket）非阻塞接收消息；收到 `publisherExecute` 后反序列化，并交给内部的 `NaiveRunner` 执行。

#### Step 1: ZMQ 层接收与反序列化

```cpp
// dtorch/core/runner/remote/remote_runner.cc:67
void RemoteRunner::ProcessSubscriberExecuteMessage(const std::string& serializedData) {
    std::vector<core::OperatorSerializationPack> opPacks;
    std::vector<uintptr_t> noHoldOperandPtrs;

    // Boost BinaryIArchive 反序列化
    std::stringstream ss(serializedData, std::ios::in | std::ios::binary);
    external::boost::BinaryIArchive bia(ss);
    bia >> opPacks;
    bia >> noHoldOperandPtrs;
    DDebugAssert(opPacks.size() + noHoldOperandPtrs.size() > 0);

    ExecuteSerialization(opPacks, noHoldOperandPtrs);
}
```

#### Step 2: 重建 Operator 并执行

```cpp
// dtorch/core/runner/remote/remote_runner.cc:79
void RemoteRunner::ExecuteSerialization(
    const std::vector<OperatorSerializationPack>& opPacks,
    const std::vector<uintptr_t>& uintNoHoldOperands) {

    std::vector<std::shared_ptr<Operator>> ops;
    for (const auto& opPack : opPacks) {
        // Step 2a: 通过 uintInputOperands 查找已注册的 Operand
        OperandArray inputOperands;
        for (auto it : opPack.uintInputOperands) {
            DAlwaysAssert(mOperandMap.count(it) > 0);
            inputOperands.push_back(mOperandMap[it]);
        }

        // Step 2b: 通过 OperatorFactory 重建 Operator
        std::shared_ptr<Operator> op =
            OperatorFactory::GetSingleton().NewOperatorOrThrow(
                opPack.opParam, inputOperands, opPack.uniqueId);

        // Step 2c: 将输出 Operand 注册到 mOperandMap
        OperandArray outputOperands = op->GetOutputOperands();
        DAlwaysAssert(outputOperands.size() == opPack.uintOutputOperands.size());
        for (size_t i = 0; i < outputOperands.size(); i++) {
            mOperandMap[opPack.uintOutputOperands[i]] = outputOperands[i];
        }
        ops.push_back(op);
    }

    // Step 2d: 收集 noHoldOperands，完成后从 map 移除
    std::vector<const Operand*> noHoldOperands;
    for (auto it : uintNoHoldOperands) {
        DAlwaysAssert(mOperandMap.count(it) > 0);
        noHoldOperands.push_back(mOperandMap[it].get());
        mOperandMap.erase(it);
    }

    // Step 2e: 提交给 NaiveRunner 执行
    Execute(ops, noHoldOperands);  // → mNaiveRunner.Execute()
}
```

**反序列化流程图**：

```
RemoteRunnerSubscriber.Get()  (ZMQ SUB, 3-frame multipart)
  │
  ├─ Frame 0: messageId     ──→  校验消息顺序（id == 上一次 + 1）
  ├─ Frame 1: "publisherExecute"
  └─ Frame 2: binary data   ──→  BinaryIArchive  ──→  OperatorSerializationPack[]
                                                              │
                                    ┌─────────────────────────┘
                                    ▼
                           RemoteRunner::ExecuteSerialization()
                                    │
                  ┌─────────────────┼─────────────────┐
                  ▼                 ▼                  ▼
          mOperandMap 查找   OperatorFactory    mOperandMap 注册
          输入 Operand      重建 Operator      输出 Operand
                  │                 │                  │
                  └─────────────────┼──────────────────┘
                                    ▼
                          NaiveRunner::Execute(ops, noHoldOperands)
```


### 3.3 Operand 指针映射机制

跨进程序列化的核心挑战在于重建计算图的拓扑连接。DTorch 的方案：

```
Controller 进程                           Worker 进程
───────────────                          ─────────────
Operand* (0x7f...128)  ── uintptr_t ──→  mOperandMap[0x7f...128] = shared_ptr<Operand>
Operand* (0x7f...256)  ── uintptr_t ──→  mOperandMap[0x7f...256] = shared_ptr<Operand>
```

- **发送端**：将 Operand 裸指针 `reinterpret_cast` 为 `uintptr_t`，存入 `uintInputOperands` / `uintOutputOperands`
- **接收端**：用 `uintptr_t` 作为 key 在 `mOperandMap` 中查找/注册 Operand
- **执行顺序保证**：ZMQ PUB-SUB 按序传递消息，Worker 端按序执行，保证当 Operator B 引用 Operator A 的输出 Operand 时，该 Operand 已经在 `mOperandMap` 中注册

## 4. 序列化格式总览

```
┌─────────────────────────────────────────────────────────────┐
│  BinaryOArchive 输出 (std::stringstream binary)              │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ vector<OperatorSerializationPack> opPacks             │    │
│  │                                                       │    │
│  │  [0] OperatorSerializationPack {                      │    │
│  │        opName: "linear_0"                             │    │
│  │        uniqueId: 1                                    │    │
│  │        opParam: ConvParam {                           │    │
│  │          mOpType: kConv           ← OpParam 基类      │    │
│  │          dilations: [1, 1]        ← ConvParam 字段    │    │
│  │          kernelSize: [3, 3]                           │    │
│  │          ...                                          │    │
│  │        }                                              │    │
│  │        uintInputOperands: [0x7f...128]                │    │
│  │        uintOutputOperands: [0x7f...256]               │    │
│  │      }                                                │    │
│  │  [1] OperatorSerializationPack {                      │    │
│  │        opName: "relu_0"                               │    │
│  │        uniqueId: 2                                    │    │
│  │        opParam: ActivationParam {                     │    │
│  │          mOpType: kActivation     ← OpParam 基类      │    │
│  │          activationKind: kRelu    ← ActivationParam   │    │
│  │        }                                              │    │
│  │        uintInputOperands: [0x7f...256]  ← 引用上一个  │    │
│  │        uintOutputOperands: [0x7f...512]               │    │
│  │      }                                                │    │
│  │  ...                                                  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ vector<uintptr_t> noHoldOperandPtrs                   │    │
│  │  [0x7f...512, ...]                                    │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 5. 相关文件索引

| 文件 | 作用 |
|---|---|
| `dtorch/external/boost/boost_serialization.h` | Boost.Serialization 封装（BinaryOArchive / BinaryIArchive / std::optional 支持） |
| `dtorch/external/boost/boost_serialization_torch.h` | torch::Tensor / torch::Generator 的序列化适配 |
| `dtorch/api/cpp/serialization.h` | `Serialization = boost::serialization::access` 别名 |
| `dtorch/api/cpp/shape.h` | Shape 序列化 |
| `dtorch/api/cpp/device.h` | Device / DeviceKey 序列化 |
| `dtorch/api/cpp/simple_array.h` | SimpleArray 序列化 |
| `dtorch/api/cpp/distributed_spec.h` | DeviceMesh / Placement / PlacementSeq 序列化 |
| `dtorch/api/cpp/scalar.h` | Scalar 序列化 |
| `dtorch/core/operators/operator_param.h` | OpParam 基类与 DTORCH_FOREACH_OPERATOR_TYPE 宏 |
| `dtorch/core/operators/operator_serialization_pack.h` | OperatorSerializationPack 定义与多态序列化模板 |
| `dtorch/core/operators/operator_serialization_pack.cc` | ToString 实现 |
| `dtorch/core/operators/operator.h` / `operator.cc` | Operator::GetOperatorSerializationPack() |
| `dtorch/core/operators/standard/*.h` | 各算子 *Param 派生类的序列化实现 |
| `dtorch/external/zmq/remote_runner_publisher.cc` | 序列化发送端 |
| `dtorch/external/zmq/remote_runner_publisher.h` | RemoteRunnerPublisher 接口 |
| `dtorch/external/zmq/remote_runner_subscriber.cc` / `.h` | ZMQ SUB 接收端（RemoteRunnerSubscriber） |
| `dtorch/core/runner/remote/remote_runner.cc` / `.h` | 反序列化后 Operator 重建与执行（RemoteRunner） |
| `dtorch/tests/test_serialization.cc` | 序列化单元测试 |
