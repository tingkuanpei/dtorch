/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "greeter_interface.h"

#include <future>

#include "dtorch/common/logging.h"
#include "dtorch/external/rpc/rpc_server_imp.h"
#ifdef __GNUC__
#pragma GCC diagnostic push
// Disable warning: 'class google::base::CheckOpMessageBuilder' has pointer data members
#pragma GCC diagnostic ignored "-Weffc++"
#endif
#include "dtorch/external/rpc/proto/greeter.grpc.pb.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

// Reference: https://github.com/grpc/grpc/blob/v1.78.1/examples/cpp/route_guide/route_guide_callback_server.cc
//            https://github.com/grpc/grpc/blob/v1.78.1/examples/cpp/route_guide/route_guide_callback_client.cc

namespace dtorch {
namespace external {
namespace rpc {

// GreeterServer

class GreeterServiceImpl final : public greeter::Greeter::CallbackService {
public:
    grpc::ServerUnaryReactor* SayGreeter(grpc::CallbackServerContext* context, const ::greeter::GreeterRequest* request,
                                         greeter::GreeterReply* reply) override {
        std::string prefix("Hello ");
        reply->set_message(prefix + request->name());
        auto* reactor = context->DefaultReactor();
        reactor->Finish(grpc::Status::OK);
        return reactor;
    }
};

GreeterServer::GreeterServer(const std::string& address) : RpcServer(address) {
    mImplPtr->service = std::make_shared<GreeterServiceImpl>();
    BuildAndStartInBackgroundThread();
}

// GreeterClient

struct GreeterClient::Impl {
    Impl() : stub() {}

    ~Impl() = default;

    DTORCH_DISABLE_COPY_AND_MOVE(Impl);

    std::unique_ptr<greeter::Greeter::Stub> stub;
};

GreeterClient::GreeterClient(const std::string& address) : mImplPtr(std::make_shared<GreeterClient::Impl>()) {
    mImplPtr->stub = greeter::Greeter::NewStub(grpc::CreateChannel(address, grpc::InsecureChannelCredentials()));
};

std::string GreeterClient::SyncSayGreeter(const std::string& user) {
    greeter::GreeterRequest request;
    request.set_name(user);
    greeter::GreeterReply reply;
    grpc::ClientContext context;
    ClientContextSetDeadline(context);
    RpcCheckStatus(mImplPtr->stub->SayGreeter(&context, request, &reply));
    return reply.message();
}

std::future<std::string> GreeterClient::AsyncSayGreeter(const std::string& user) {
    std::shared_ptr<grpc::ClientContext> context = std::make_shared<grpc::ClientContext>();
    ClientContextSetDeadline(*context);
    greeter::GreeterRequest request;
    request.set_name(user);
    std::shared_ptr<greeter::GreeterReply> reply = std::make_shared<greeter::GreeterReply>();
    auto promiseStr = std::make_shared<std::promise<std::string>>();
    std::future<std::string> futureStr = promiseStr->get_future();

    mImplPtr->stub->async()->SayGreeter(context.get(), &request, reply.get(),
                                        [context, reply, promiseStr](grpc::Status status) {
                                            RpcCheckStatus(status);
                                            promiseStr->set_value(reply->message());
                                        });

    return futureStr;
}

}  // namespace rpc
}  // namespace external
}  // namespace dtorch
