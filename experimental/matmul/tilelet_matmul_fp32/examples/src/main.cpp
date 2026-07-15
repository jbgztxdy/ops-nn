/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file main.cpp
 * \brief
 */
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "acl/acl.h"
#include "common.h"
#include "op_runner.h"

bool g_isDevice = false;
int deviceId = 0;

namespace {
bool GetEnvFlag(const char* name, bool defaultValue = false)
{
    const char* value = std::getenv(name);
    return value == nullptr ? defaultValue : std::strtoll(value, nullptr, 10) != 0;
}

bool IsBf16()
{
    const char* value = std::getenv("TILELET_DTYPE");
    return value != nullptr && std::string(value) == "bf16";
}
} // namespace

OperatorDesc CreateOpDesc()
{
    // define operator
    bool transposeX2 = GetEnvFlag("TILELET_TRANSPOSE_X2");
    bool useBias = GetEnvFlag("TILELET_USE_BIAS", true);
    std::vector<int64_t> shapeA{256, 64};
    std::vector<int64_t> shapeB = transposeX2 ? std::vector<int64_t>{512, 64} : std::vector<int64_t>{64, 512};
    std::vector<int64_t> shapeBias{512};
    std::vector<int64_t> shapeC{256, 512};
    aclDataType dataType = IsBf16() ? ACL_BF16 : ACL_FLOAT;
    aclFormat format = ACL_FORMAT_ND;
    OperatorDesc opDesc;
    opDesc.AddInputTensorDesc(dataType, shapeA.size(), shapeA.data(), format);
    opDesc.AddInputTensorDesc(dataType, shapeB.size(), shapeB.data(), format);
    if (useBias) {
        opDesc.AddInputTensorDesc(dataType, shapeBias.size(), shapeBias.data(), format);
    }
    opDesc.AddOutputTensorDesc(dataType, shapeC.size(), shapeC.data(), format);
    return opDesc;
}

bool SetInputData(OpRunner& runner)
{
    size_t fileSize = 0;
    ReadFile("../input/input_a.bin", fileSize, runner.GetInputBuffer<void>(0), runner.GetInputSize(0));
    ReadFile("../input/input_b.bin", fileSize, runner.GetInputBuffer<void>(1), runner.GetInputSize(1));
    if (runner.NumInputs() > 2) {
        ReadFile("../input/input_bias.bin", fileSize, runner.GetInputBuffer<void>(2), runner.GetInputSize(2));
    }
    INFO_LOG("Set input success");
    return true;
}

bool ProcessOutputData(OpRunner& runner)
{
    WriteFile("../output/output_z.bin", runner.GetOutputBuffer<void>(0), runner.GetOutputSize(0));
    INFO_LOG("Write output success");
    return true;
}

void DestroyResource()
{
    bool flag = false;
    if (aclrtResetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Reset device %d failed", deviceId);
        flag = true;
    }
    INFO_LOG("Reset Device success");
    if (aclFinalize() != ACL_SUCCESS) {
        ERROR_LOG("Finalize acl failed");
        flag = true;
    }
    if (flag) {
        ERROR_LOG("Destroy resource failed");
    } else {
        INFO_LOG("Destroy resource success");
    }
}

bool InitResource()
{
    std::string output = "../output";
    if (access(output.c_str(), 0) == -1) {
        int ret = mkdir(output.c_str(), 0700);
        if (ret == 0) {
            INFO_LOG("Make output directory successfully");
        } else {
            ERROR_LOG("Make output directory fail");
            return false;
        }
    }

    if (aclInit(nullptr) != ACL_SUCCESS) {
        ERROR_LOG("acl init failed");
        return false;
    }

    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Set device failed. deviceId is %d", deviceId);
        (void)aclFinalize();
        return false;
    }
    INFO_LOG("Set device[%d] success", deviceId);

    // runMode is ACL_HOST which represents app is running in host
    // runMode is ACL_DEVICE which represents app is running in device
    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
        ERROR_LOG("Get run mode failed");
        DestroyResource();
        return false;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    INFO_LOG("Get RunMode[%d] success", runMode);

    return true;
}

bool RunOp()
{
    // create op desc
    OperatorDesc opDesc = CreateOpDesc();

    // create Runner
    OpRunner opRunner(&opDesc);
    if (!opRunner.Init()) {
        ERROR_LOG("Init OpRunner failed");
        return false;
    }

    // Load inputs
    if (!SetInputData(opRunner)) {
        ERROR_LOG("Set input data failed");
        return false;
    }

    // Run op
    if (!opRunner.RunOp()) {
        ERROR_LOG("Run op failed");
        return false;
    }

    // process output data
    if (!ProcessOutputData(opRunner)) {
        ERROR_LOG("Process output data failed");
        return false;
    }

    INFO_LOG("Run op success");
    return true;
}

int main(int argc, char** argv)
{
    if (!InitResource()) {
        ERROR_LOG("Init resource failed");
        return FAILED;
    }
    INFO_LOG("Init resource success");

    if (!RunOp()) {
        DestroyResource();
        return FAILED;
    }

    DestroyResource();

    return SUCCESS;
}
