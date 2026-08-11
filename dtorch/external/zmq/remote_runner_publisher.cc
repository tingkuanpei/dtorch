/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "remote_runner_publisher.h"

#include "dtorch/common/debug.h"
#include "dtorch/core/operators/operator_serialization_pack.h"
#include "dtorch/external/boost/boost_serialization.h"
#include "dtorch/external/zmq/zmq.h"

namespace dtorch {
namespace external {
namespace zmq {

struct RemoteRunnerPublisher::Impl {
    Impl(const std::string& address) : address(address), ctx(), publisher(ctx, ::zmq::socket_type::pub) {
        // Disable message queue limit
        publisher.set(::zmq::sockopt::sndhwm, 0);
        publisher.bind(address);
    }

    ~Impl() = default;

    DTORCH_DISABLE_COPY_AND_MOVE(Impl);
    std::string address;
    ::zmq::context_t ctx;
    ::zmq::socket_t publisher;
};

RemoteRunnerPublisher::RemoteRunnerPublisher(const std::string& address) : mImplPtr(std::make_shared<Impl>(address)) {}

RemoteRunnerPublisher::~RemoteRunnerPublisher() { SendDestroy(); }

void RemoteRunnerPublisher::Execute(const std::vector<std::shared_ptr<core::Operator>>& ops,
                                    const std::vector<const core::Operand*>& noHoldOperands) {
    std::vector<core::OperatorSerializationPack> opPacks;
    std::vector<uintptr_t> noHoldOperandPtrs;
    for (const auto& op : ops) {
        opPacks.push_back(op->GetOperatorSerializationPack());
    }
    for (const auto& operand : noHoldOperands) {
        noHoldOperandPtrs.push_back(reinterpret_cast<uintptr_t>(operand));
    }

    std::stringstream ss(std::ios::out | std::ios::binary);
    boost::BinaryOArchive boa(ss);
    boa << opPacks;
    boa << noHoldOperandPtrs;
    std::string serializedData = ss.str();

    int64_t messageId = PublishMessageIdManager::GetSingleton().GetIdAndIncrement(mImplPtr->address);
    const std::array<::zmq::const_buffer, 3> send_msgs = {::zmq::buffer(std::to_string(messageId)),
                                                          ::zmq::buffer(RemoteRunnerPublisher::kExecuteStr),
                                                          ::zmq::buffer(serializedData.data(), serializedData.size())};
    SendMultipart(mImplPtr->publisher, send_msgs);
}

void RemoteRunnerPublisher::SendDestroy() {
    int64_t messageId = PublishMessageIdManager::GetSingleton().GetIdAndIncrement(mImplPtr->address);
    const std::array<::zmq::const_buffer, 2> send_msgs = {::zmq::buffer(std::to_string(messageId)),
                                                          ::zmq::buffer(RemoteRunnerPublisher::kDestroyStr)};
    SendMultipart(mImplPtr->publisher, send_msgs);
}

}  // namespace zmq
}  // namespace external
}  // namespace dtorch
