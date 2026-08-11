/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "global_option.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "dtorch/common/debug.h"
#include "dtorch/common/environment_variable.h"
#include "dtorch/common/logging.h"

namespace dtorch {
namespace core {

GlobalOption::GlobalOption()
    : mCommTimeoutSecond(600),
      mGrpcTimeoutSecond(3),
      mZmqTimeoutSecond(3),
      mDTensorInSameDevice(false),
      mPerDevicePerProcess(false),
      mNumGpuWhenEnableDtensorInSameDevice(8),
      mValidateKernelInputOutput(false) {
    mCommTimeoutSecond = GetEnvVarAsInt("DTORCH_COMM_TIMEOUT_SECOND", mCommTimeoutSecond);
    if (mCommTimeoutSecond <= 0) {
        DLogFatal() << "Environment variable DTORCH_COMM_TIMEOUT_SECOND must be greater than 0";
    }
    mGrpcTimeoutSecond = GetEnvVarAsInt("DTORCH_GRPC_TIMEOUT_SECOND", mGrpcTimeoutSecond);
    if (mGrpcTimeoutSecond <= 0) {
        DLogFatal() << "Environment variable DTORCH_GRPC_TIMEOUT_SECOND must be greater than 0";
    }
    mZmqTimeoutSecond = GetEnvVarAsInt("DTORCH_ZMQ_TIMEOUT_SECOND", mZmqTimeoutSecond);
    if (mZmqTimeoutSecond <= 0) {
        DLogFatal() << "Environment variable DTORCH_ZMQ_TIMEOUT_SECOND must be greater than 0";
    }
    mDTensorInSameDevice = GetEnvVarAsBool("DTORCH_DTENSOR_IN_SAME_DEVICE", mDTensorInSameDevice);
    mPerDevicePerProcess = GetEnvVarAsBool("DTORCH_PER_DEVICE_PER_PROCESS", mPerDevicePerProcess);
    mNumGpuWhenEnableDtensorInSameDevice =
        GetEnvVarAsInt("DTORCH_NUM_GPU_WHEN_ENABLE_DTENSOR_IN_SAME_DEVICE", mNumGpuWhenEnableDtensorInSameDevice);
    if (mNumGpuWhenEnableDtensorInSameDevice <= 0) {
        DLogFatal() << "Environment variable DTORCH_NUM_GPU_WHEN_ENABLE_DTENSOR_IN_SAME_DEVICE must be greater than 0";
    }
    mValidateKernelInputOutput = GetEnvVarAsBool("DTORCH_VALIDATE_KERNEL_INPUT_OUTPUT", mValidateKernelInputOutput);
}

GlobalOption::GlobalOption(DeserializeTag)
    : mCommTimeoutSecond(0),
      mGrpcTimeoutSecond(0),
      mZmqTimeoutSecond(0),
      mDTensorInSameDevice(false),
      mPerDevicePerProcess(false),
      mNumGpuWhenEnableDtensorInSameDevice(0),
      mValidateKernelInputOutput(false) {}

void GlobalOption::GlobalOptionInitFromString(const std::string& serializedData) {
    std::istringstream iss(serializedData);
    boost::archive::text_iarchive ia(iss);
    GlobalOption mainOpt(DeserializeTag{});
    ia >> mainOpt;
    GetSingleton() = mainOpt;
}

std::string GlobalOption::SerializeToString() const {
    std::ostringstream oss;
    boost::archive::text_oarchive oa(oss);
    oa << *this;
    return oss.str();
}

GlobalOption& GlobalOption::operator=(const GlobalOption& other) {
    mCommTimeoutSecond = other.mCommTimeoutSecond;
    mGrpcTimeoutSecond = other.mGrpcTimeoutSecond;
    mZmqTimeoutSecond = other.mZmqTimeoutSecond;
    mDTensorInSameDevice = other.mDTensorInSameDevice;
    mPerDevicePerProcess = other.mPerDevicePerProcess;
    mNumGpuWhenEnableDtensorInSameDevice = other.mNumGpuWhenEnableDtensorInSameDevice;
    mValidateKernelInputOutput = other.mValidateKernelInputOutput;
    return *this;
}

}  // namespace core
}  // namespace dtorch
