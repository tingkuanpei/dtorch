/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "ip_address.h"

#include <stdexcept>
#include <string>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace dtorch {
namespace {

using SocketHandle = int;
constexpr SocketHandle kInvalidSocket = -1;

void CloseSocket(SocketHandle socketFd) { close(socketFd); }

std::string GetNodeIp() {
    const std::string kRemoteIp = "8.8.8.8";
    constexpr uint16_t kRemotePort = 80;

    SocketHandle socketFd = socket(AF_INET, SOCK_DGRAM, 0);
    if (socketFd == kInvalidSocket) {
        throw std::runtime_error("create UDP socket failed");
    }

    sockaddr_in remoteAddress{};
    remoteAddress.sin_family = AF_INET;
    remoteAddress.sin_port = htons(kRemotePort);
    if (inet_pton(AF_INET, kRemoteIp.c_str(), &remoteAddress.sin_addr) <= 0) {
        CloseSocket(socketFd);
        throw std::runtime_error("parse remote IP failed");
    }

    if (connect(socketFd, reinterpret_cast<const sockaddr*>(&remoteAddress), sizeof(remoteAddress)) != 0) {
        CloseSocket(socketFd);
        throw std::runtime_error("connect UDP socket failed");
    }

    sockaddr_in localAddress{};
    socklen_t localAddressLen = sizeof(localAddress);
    if (getsockname(socketFd, reinterpret_cast<sockaddr*>(&localAddress), &localAddressLen) != 0) {
        CloseSocket(socketFd);
        throw std::runtime_error("getsockname failed");
    }

    char ipBuffer[INET_ADDRSTRLEN] = {0};
    if (inet_ntop(AF_INET, &localAddress.sin_addr, ipBuffer, sizeof(ipBuffer)) == nullptr) {
        CloseSocket(socketFd);
        throw std::runtime_error("convert local IP to string failed");
    }

    CloseSocket(socketFd);
    return std::string(ipBuffer);
}

bool CheckPortStatus(const std::string& ip, int64_t port) {
    if (port <= 0 || port >= 65536) {
        throw std::invalid_argument("port invalid, MUST between 1 and 65535");
    }

    SocketHandle socketFd = socket(AF_INET, SOCK_STREAM, 0);
    if (socketFd == kInvalidSocket) {
        return false;
    }

    int reuseAddress = 1;
    setsockopt(socketFd, SOL_SOCKET, SO_REUSEADDR, &reuseAddress, sizeof(reuseAddress));

    sockaddr_in localAddress{};
    localAddress.sin_family = AF_INET;
    localAddress.sin_port = htons(static_cast<uint16_t>(port));
    if (inet_pton(AF_INET, ip.c_str(), &localAddress.sin_addr) <= 0) {
        CloseSocket(socketFd);
        return false;
    }

    const bool isAvailable =
        bind(socketFd, reinterpret_cast<const sockaddr*>(&localAddress), sizeof(localAddress)) == 0;
    CloseSocket(socketFd);
    return isAvailable;
}

}  // namespace

bool CheckStringIsValidAddress(const std::string& str) {
    // Expected format: ip:port
    const auto colonPos = str.rfind(':');
    if (colonPos == std::string::npos || colonPos == 0 || colonPos == str.size() - 1) {
        return false;
    }

    const std::string ip = str.substr(0, colonPos);
    const std::string portStr = str.substr(colonPos + 1);

    // Validate IP via inet_pton
    struct in_addr ipv4Addr;
    if (inet_pton(AF_INET, ip.c_str(), &ipv4Addr) != 1) {
        return false;
    }

    // Validate port: must be numeric and in range [1, 65535]
    try {
        int port = std::stoi(portStr);
        if (port < 1 || port > 65535) {
            return false;
        }
    } catch (const std::exception&) {
        return false;
    }

    return true;
}

std::string GetValidNodeAddress(int64_t startPort, int64_t endPort) {
    const std::string nodeIp = GetNodeIp();

    for (int64_t port = startPort; port < endPort; port++) {
        if (CheckPortStatus(nodeIp, port)) {
            return nodeIp + ":" + std::to_string(port);
        }
    }

    throw std::runtime_error("node ip: " + nodeIp + ", port from " + std::to_string(startPort) + " to " +
                             std::to_string(endPort) + " invalid");
}

}  // namespace dtorch
