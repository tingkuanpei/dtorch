/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include <chrono>
#include <iostream>
#include <thread>

#include "dtorch/common/argument_parser.h"
#include "dtorch/common/process/sub_process.h"
#include "dtorch/external/zmq/zmq.h"
#include "test.h"

using dtorch::external::zmq::GetMsgAsString;
using dtorch::external::zmq::GetZmqTimeoutMilliSecond;
using dtorch::external::zmq::RecvMultipart;
using dtorch::external::zmq::SendMultipart;

TEST(ZeroMQTest, SimpleTest) {
    zmq::context_t ctx;
    zmq::socket_t rep(ctx, zmq::socket_type::rep);
    zmq::socket_t req(ctx, zmq::socket_type::req);

    rep.set(zmq::sockopt::rcvtimeo, GetZmqTimeoutMilliSecond());
    rep.set(zmq::sockopt::sndtimeo, GetZmqTimeoutMilliSecond());
    req.set(zmq::sockopt::rcvtimeo, GetZmqTimeoutMilliSecond());
    req.set(zmq::sockopt::sndtimeo, GetZmqTimeoutMilliSecond());

    rep.bind("tcp://127.0.0.1:*");
    const std::string endpoint = rep.get(zmq::sockopt::last_endpoint);

    const std::string requestPayload = "hello_req_rep";
    const std::string replyPayload = "ack:" + requestPayload;

    std::thread server([&]() {
        std::vector<zmq::message_t> recvMsgs;
        RecvMultipart(rep, recvMsgs);

        const std::string got = GetMsgAsString(recvMsgs[0]);
        EXPECT_EQ(got, requestPayload);

        const std::array<zmq::const_buffer, 1> send_msgs = {zmq::buffer(replyPayload)};
        SendMultipart(rep, send_msgs);
    });

    req.connect(endpoint);

    const std::array<zmq::const_buffer, 1> send_msgs = {zmq::buffer(requestPayload)};
    SendMultipart(req, send_msgs);

    std::vector<zmq::message_t> reply_msgs;
    RecvMultipart(req, reply_msgs);
    const std::string gotReply = GetMsgAsString(reply_msgs[0]);
    EXPECT_EQ(gotReply, replyPayload);

    server.join();
}

TEST(ZeroMQTest, SinglePublisherDoubleSubscribers) {
    zmq::context_t ctx;
    zmq::socket_t pub(ctx, zmq::socket_type::pub);
    zmq::socket_t sub1(ctx, zmq::socket_type::sub);
    zmq::socket_t sub2(ctx, zmq::socket_type::sub);

    pub.bind("tcp://127.0.0.1:*");
    const std::string endpoint = pub.get(zmq::sockopt::last_endpoint);

    sub1.set(zmq::sockopt::subscribe, "");
    sub2.set(zmq::sockopt::subscribe, "");
    sub1.set(zmq::sockopt::rcvtimeo, GetZmqTimeoutMilliSecond());
    sub2.set(zmq::sockopt::rcvtimeo, GetZmqTimeoutMilliSecond());

    sub1.connect(endpoint);
    sub2.connect(endpoint);

    // Avoid the PUB/SUB slow-joiner issue by waiting for subscriptions.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    const std::string payload = "hello_two_subscribers";
    const std::array<zmq::const_buffer, 1> send_msgs = {zmq::buffer(payload)};
    SendMultipart(pub, send_msgs);

    std::vector<zmq::message_t> recv1;
    std::vector<zmq::message_t> recv2;
    RecvMultipart(sub1, recv1);
    RecvMultipart(sub2, recv2);

    const std::string got1 = GetMsgAsString(recv1[0]);
    const std::string got2 = GetMsgAsString(recv2[0]);
    EXPECT_EQ(got1, payload);
    EXPECT_EQ(got2, payload);
}

TEST(ZeroMQTest, InterprocessSinglePublisherSingleSubscriber) {
    const auto &parser = dtorch::ArgumentParser::GetSingleton();
    if (parser.HasOption("child")) {
        // Child Process
        zmq::context_t ctx;
        zmq::socket_t sub(ctx, zmq::socket_type::sub);

        sub.set(zmq::sockopt::subscribe, "");
        sub.set(zmq::sockopt::rcvtimeo, GetZmqTimeoutMilliSecond());
        sub.connect(parser.OptionValue("endpoint"));

        std::vector<zmq::message_t> recvMsgs;
        RecvMultipart(sub, recvMsgs);

        const std::string got = GetMsgAsString(recvMsgs[0]);
        EXPECT_EQ(got, "hello_interprocess_pubsub");
        return;
    } else {
        // Parent Process
        zmq::context_t ctx;
        zmq::socket_t pub(ctx, zmq::socket_type::pub);
        pub.bind("tcp://127.0.0.1:*");
        const std::string endpoint = pub.get(zmq::sockopt::last_endpoint);

        const std::string exeCmd = parser.ProgramName() +
                                   " --gtest_filter=ZeroMQTest.InterprocessSinglePublisherSingleSubscriber --child" +
                                   " --endpoint=" + endpoint;
        dtorch::SubProcess subProcess(exeCmd);

        // Give subscriber process enough time to connect before first publish.
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        const std::string payload = "hello_interprocess_pubsub";
        const std::array<zmq::const_buffer, 1> send_msgs = {zmq::buffer(payload)};
        SendMultipart(pub, send_msgs);
    }
}
