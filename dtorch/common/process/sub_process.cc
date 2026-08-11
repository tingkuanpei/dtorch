/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "sub_process.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include <boost/asio.hpp>
#include <boost/process.hpp>
#include <boost/process/environment.hpp>
#include <boost/process/shell.hpp>

#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/common/string.h"

namespace dtorch {

// CUDA initialization error after fork:
// https://stackoverflow.com/questions/22950047/cuda-initialization-error-after-fork

struct SubProcess::Impl {
    Impl() : process() {}

    ~Impl() {
        process->wait();
        DAlwaysAssertMsg(process->exit_code() == 0, "SubProcess exit with error!!!");
    }

    std::unique_ptr<boost::process::child> process;
};

SubProcess::SubProcess() : mImplPtr() {}

SubProcess::SubProcess(const std::string& exeCmd, const std::unordered_map<std::string, std::string>& envs)
    : mImplPtr(std::make_shared<Impl>()) {
    try {
        // The child process needs to inherit all environment variables from the parent process.
        boost::process::environment currentEnv = boost::this_process::environment();
        for (const auto& [key, value] : envs) {
            currentEnv[key] = value;
        }
        mImplPtr->process = std::make_unique<boost::process::child>(exeCmd, currentEnv);
    } catch (std::exception& e) {
        // If error msg: No such file or directory, it means the executable file does not exist.
        // Please check PATH environment variable.
        std::stringstream ss;
        ss << "Start process failed, execute command: " << exeCmd << ", error msg: " << e.what()
           << " . Please check PATH environment variable.";
        DLogFatal() << ss.str();
    }
}

SubProcess::~SubProcess() = default;

}  // namespace dtorch
