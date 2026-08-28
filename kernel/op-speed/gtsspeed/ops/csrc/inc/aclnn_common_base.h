/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PYTORCH_NPU_ACLNN_COMMON_BASE_HPP_
#define PYTORCH_NPU_ACLNN_COMMON_BASE_HPP_

#include <dlfcn.h>
#include <vector>
#include <functional>
#include <type_traits>
#include <ATen/Tensor.h>
#include <acl/acl_base.h>

// typedef struct aclmdlRITask aclmdlRITask;
// typedef struct aclmdlRITaskParams aclmdlRITaskParams;

#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include <acl/acl_rt.h>
#include <c10/util/Exception.h>
#include <securec.h>
#include <unordered_map>
#include <mutex>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"

#define NPU_NAME_SPACE at_npu::native
#define UNUSE(name) (void)(name)

namespace c10::detail {
inline void torchCheckFail(const char* func, const char* file, uint32_t line, const std::string& msg)
{
    torchCheckFail(func, file, line, msg.c_str());
}
}

inline std::string getCurrentTimestamp()
{
    return "unknown";
}

using aclOpExecutor = struct aclOpExecutor;
using aclTensor = struct aclTensor;
using aclScalar = struct aclScalar;
using aclIntArray = struct aclIntArray;
using aclFloatArray = struct aclFloatArray;
using aclBoolArray = struct aclBoolArray;
using aclTensorList = struct aclTensorList;

using _aclCreateTensor = aclTensor *(*)(const int64_t *view_dims, uint64_t view_dims_num, aclDataType data_type,
    const int64_t *stride, int64_t offset, aclFormat format, const int64_t *storage_dims, uint64_t storage_dims_num,
    void *tensor_data);
using _aclCreateScalar = aclScalar *(*)(void *value, aclDataType data_type);
using _aclCreateIntArray = aclIntArray *(*)(const int64_t *value, uint64_t size);
using _aclCreateFloatArray = aclFloatArray *(*)(const float *value, uint64_t size);
using _aclCreateBoolArray = aclBoolArray *(*)(const bool *value, uint64_t size);
using _aclCreateTensorList = aclTensorList *(*)(const aclTensor *const *value, uint64_t size);

using _aclDestroyTensor = int (*)(const aclTensor *tensor);
using _aclDestroyScalar = int (*)(const aclScalar *scalar);
using _aclDestroyIntArray = int (*)(const aclIntArray *array);
using _aclDestroyFloatArray = int (*)(const aclFloatArray *array);
using _aclDestroyBoolArray = int (*)(const aclBoolArray *array);
using _aclDestroyTensorList = int (*)(const aclTensorList *array);

constexpr int kHashBufSize = 8192;
constexpr int kHashBufMaxSize = kHashBufSize + 1024;
extern thread_local char g_hashBuf[kHashBufSize];
extern thread_local int g_hashOffset;

// from at::ScalarType
constexpr aclDataType kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(at::ScalarType::NumOptions) + 1] = {
    ACL_UINT8, ACL_INT8, ACL_INT16, ACL_INT32, ACL_INT64, ACL_FLOAT16, ACL_FLOAT, ACL_DOUBLE, ACL_DT_UNDEFINED,
    ACL_COMPLEX64, ACL_COMPLEX128, ACL_BOOL, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED, ACL_BF16,
    ACL_DT_UNDEFINED, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED
};

using OpApiFunc = int (*)(void *, uint64_t, aclOpExecutor *, const aclrtStream);
#define GET_OP_API_FUNC(apiName) reinterpret_cast<_##apiName>(GetOpApiFuncAddr(#apiName))

struct ApiHandleManager {
public:
    static ApiHandleManager& GetInstance()
    {
        static ApiHandleManager instance;
        return instance;
    }

    ~ApiHandleManager()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        apiHandles_.clear();
        for (auto& pair : libHandles_) {
            ::dlclose(pair.second);
        }
        libHandles_.clear();
    }

    void* Get(const std::string& apiName)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = apiHandles_.find(apiName);
        if (it != apiHandles_.end()) {
            return it->second;
        }

        auto apiHandle = GetOpApiFuncAddr(apiName);
        if (apiHandle != nullptr) {
            apiHandles_[apiName] = apiHandle;
        }
        return apiHandle;
    }

    static std::string GetOpApiLibName()
    {
        return "libopapi.so";
    }
private:
    static std::string GetCustOpApiLibName()
    {
        return "libcust_opapi.so";
    }

    static void* GetOpApiFuncAddrInLib(void *handler, const std::string& libName, const std::string& apiName)
    {
        auto funcAddr = ::dlsym(handler, apiName.c_str());
        if (funcAddr != nullptr) {
            ASCEND_LOGI("dlsym %s from %s success", apiName.c_str(), libName.c_str());
            return funcAddr;
        }

        ASCEND_LOGW("dlsym %s from %s failed, error:%s.", apiName.c_str(), libName.c_str(), dlerror());
        return nullptr;
    }

    void* GetOpApiLibHandler(const std::string& libName)
    {
        auto iter = libHandles_.find(libName);
        if (iter != libHandles_.end()) {
            return iter->second;
        }

        auto handle = ::dlopen(libName.c_str(), RTLD_LAZY);
        if (handle == nullptr) {
            ASCEND_LOGW("dlopen %s failed, error:%s", libName.c_str(), dlerror());
            std::cout << "dlopen " << libName << " failed, error:" << dlerror() << std::endl;
            return nullptr;
        }
        libHandles_[libName] = handle;
        ASCEND_LOGW("dlopen %s success", libName.c_str());
        // std::cout << "dlopen " << libName << " success" << std::endl;
        return handle;
    }

    inline void* GetFuncAddr(const std::string& soName, const std::string& apiName)
    {
        static auto opApiHandler = GetOpApiLibHandler(soName);
        if (opApiHandler != nullptr) {
            auto funcAddr = GetOpApiFuncAddrInLib(opApiHandler, soName, apiName);
            if (funcAddr != nullptr) {
                return funcAddr;
            }
        }
        return nullptr;
    }

    inline void* GetOpApiFuncAddr(const std::string& apiName)
    {
        auto customOppPathChar = std::getenv("GTS_CUSTOM_OPP_PATH");
        if (customOppPathChar != nullptr) {
            std::string customOppPath = std::string(customOppPathChar);
            auto pos = customOppPath.find(";:");
            customOppPath = customOppPath.substr(0, pos);
            auto customFuncAddr = GetFuncAddr(customOppPath + "/op_api/lib/" + GetCustOpApiLibName(), apiName);
            if (customFuncAddr != nullptr) {
                return customFuncAddr;
            }
        }

        auto funcAddr = GetFuncAddr(GetCustOpApiLibName(), apiName);
        if (funcAddr != nullptr) {
            return funcAddr;
        }

        funcAddr = GetFuncAddr(GetOpApiLibName(), apiName);
        if (funcAddr != nullptr) {
            return funcAddr;
        }

        ASCEND_LOGW("=============GetOpApiFuncAddr(%s) failed!", apiName.c_str());
        return nullptr;
    }

private:
    ApiHandleManager() = default;
    std::mutex mutex_;
    std::unordered_map<std::string, void*> libHandles_;
    std::unordered_map<std::string, void*> apiHandles_;
};

inline void *GetOpApiFuncAddr(const std::string& apiName)
{
    return ApiHandleManager::GetInstance().Get(apiName);
}

inline void *GetOpApiFuncAddr(const char* apiName)
{
    return ApiHandleManager::GetInstance().Get(apiName);
}

inline std::string GetOpApiLibName()
{
    return ApiHandleManager::GetInstance().GetOpApiLibName();
}

inline c10::Scalar ConvertTensorToScalar(const at::Tensor &tensor)
{
    c10::Scalar expScalar;
    const at::Tensor *aclInput = &tensor;
    if (aclInput->scalar_type() == at::ScalarType::Double) {
        double value = *(double *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::Long) {
        int64_t value = *(int64_t *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::Float) {
        float value = *(float *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::Int) {
        int value = *(int *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::Half) {
        c10::Half value = *(c10::Half *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::Bool) {
        int8_t value = *(int8_t *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::ComplexDouble) {
        c10::complex<double> value = *(c10::complex<double> *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::ComplexFloat) {
        c10::complex<float> value = *(c10::complex<float> *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::BFloat16) {
        c10::BFloat16 value = *(c10::BFloat16 *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else {
        ASCEND_LOGE("unsupported scalar type! ");
    }
    return expScalar;
}

inline at::Tensor CopyTensorHostToDevice(const at::Tensor &cpu_tensor)
{
    at::Tensor cpuPinMemTensor = cpu_tensor.pin_memory();
    int deviceIndex = 0;
    return cpuPinMemTensor.to(
        c10::Device(torch_npu::utils::get_npu_device_type(), deviceIndex), cpuPinMemTensor.scalar_type(), true, true);
}

inline at::Tensor CopyScalarToDevice(const c10::Scalar &cpu_scalar, at::ScalarType scalar_data_type)
{
    return CopyTensorHostToDevice(scalar_to_tensor(cpu_scalar).to(scalar_data_type));
}

#endif  // PYTORCH_NPU_HELPER_HPP_
