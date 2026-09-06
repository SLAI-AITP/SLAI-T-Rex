/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CANN_PARAM_HPP
#define CANN_PARAM_HPP
#include <type_traits>

namespace AscendC {
// 知识点：不支持模板类的模板方法特例化
struct ParamStruct {
    using Type = float;
    static constexpr uint32_t ParamCount = 64;
public:
    __aicore__ ParamStruct() = default;

    template <typename OriginType>
    __aicore__ void Init(TPipe &pipe, GM_ADDR _paramGm) {
        if (_paramGm == nullptr) {
            offset = -1;
            return;
        }
        GlobalTensor<OriginType> paramGm;
        paramGm.SetGlobalBuffer((__gm__ OriginType*)_paramGm);

        TBuf<TPosition::VECCALC> originParamBuf;
        pipe.InitBuffer(originParamBuf, ParamCount * sizeof(OriginType));

        auto originParams = originParamBuf.Get<OriginType>();
        DataCopy(originParams, paramGm, ParamCount);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        pipe.InitBuffer(paramBuf, ParamCount * sizeof(Type));
        params = paramBuf.Get<Type>();

        // 方案1：采用取值后赋值的方式
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        for (size_t i = 0; i < ParamCount; ++i) {
            Type value;
            // 知识点：支持如下的常量表达式推导
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
            if constexpr (std::is_same_v<bfloat16_t, OriginType>) {
                // 知识点：bfloat16类型转换需要调用AscendC::ToBfloat16、AscendC::ToFloat
                value = AscendC::ToFloat(originParams.GetValue(i));
            } else
#endif
            if constexpr (std::is_same_v<half, OriginType> || std::is_same_v<float, OriginType>) {
                value = static_cast<float>(originParams.GetValue(i));
            } else if constexpr (std::is_unsigned_v<OriginType>) {
                value = static_cast<int32_t>(originParams.GetValue(i));
            } else {
                value = originParams.GetValue(i);
            }

            params.SetValue(i, value);
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    }

    template <typename T>
    __aicore__ T Get() {
        if (offset == -1) {
            return {};
        }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        if constexpr (std::is_same_v<bfloat16_t, T>) {
            return AscendC::ToBfloat16(static_cast<float>(params.GetValue(offset++)));
        } else
#endif
        if constexpr (std::is_same_v<half, T> || std::is_same_v<float, T>) {
            return static_cast<T>(static_cast<float>(params.GetValue(offset++)));
        } else if constexpr (std::is_unsigned_v<T>) {
            return static_cast<int32_t>(params.GetValue(offset++));
        } else {
            return static_cast<T>(params.GetValue(offset++));
        }
    }

    __aicore__ bool IsInit()
    {
        return (offset >= 0);
    }

private:
    size_t offset = 0;
    TBuf<TPosition::VECCALC> paramBuf;
    LocalTensor<Type> params;
};

template <>
__aicore__ inline CopyRepeatParams ParamStruct::Get() {
    CopyRepeatParams copyRepeatParams;
    copyRepeatParams.dstStride = Get<uint16_t>();
    copyRepeatParams.srcStride = Get<uint16_t>();
    copyRepeatParams.dstRepeatSize = Get<uint16_t>(); // [0~8] dst 重复使用的block数
    copyRepeatParams.srcRepeatSize = Get<uint16_t>(); // [0~8] src 重复使用的block数
    return copyRepeatParams;
}

template <>
__aicore__ inline BinaryRepeatParams ParamStruct::Get() {
    BinaryRepeatParams binaryRepeatParams;
    binaryRepeatParams.blockNumber = Get<uint32_t>();
    binaryRepeatParams.dstBlkStride = Get<uint8_t>();
    binaryRepeatParams.src0BlkStride = Get<uint8_t>();
    binaryRepeatParams.src1BlkStride = Get<uint8_t>();
    binaryRepeatParams.dstRepStride = Get<uint8_t>();
    binaryRepeatParams.src0RepStride = Get<uint8_t>();
    binaryRepeatParams.src1RepStride = Get<uint8_t>();
    binaryRepeatParams.repeatStrideMode = (Get<uint32_t>() > 0);
    binaryRepeatParams.strideSizeMode = (Get<uint32_t>() > 0);
    return binaryRepeatParams;
}

template <>
__aicore__ inline DataCopyParams ParamStruct::Get() {
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = Get<uint16_t>();
    dataCopyParams.blockLen = Get<uint16_t>();
    dataCopyParams.srcStride = Get<uint16_t>();
    dataCopyParams.dstStride = Get<uint16_t>();
    return dataCopyParams;
}

template <>
__aicore__ inline DataCopyPadParams ParamStruct::Get() {
    DataCopyPadParams padParams;
    padParams.isPad = (Get<uint16_t>() > 0);
    padParams.leftPadding = Get<uint8_t>();
    padParams.rightPadding = Get<uint8_t>();
    padParams.paddingValue = Get<uint64_t>();
    return padParams;
}

// url: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/API/ascendcopapi/atlasascendc_api_07_0012.html
template <>
__aicore__ inline UnaryRepeatParams ParamStruct::Get() {
    UnaryRepeatParams unaryRepeatParams;
    unaryRepeatParams.blockNumber = (Get<uint32_t>());
    unaryRepeatParams.dstBlkStride = Get<uint16_t>();
    unaryRepeatParams.srcBlkStride = Get<uint16_t>();
    unaryRepeatParams.dstRepStride   = Get<uint8_t>();
    unaryRepeatParams.srcRepStride   = Get<uint8_t>();
    unaryRepeatParams.repeatStrideMode = (Get<uint32_t>() > 0);
    unaryRepeatParams.strideSizeMode  = (Get<uint32_t>() > 0);
    unaryRepeatParams.halfBlock = (Get<uint32_t>() > 0); // 默认0
    return unaryRepeatParams;
}

} // end namespace AscendC
#endif  // CANN_PARAM_HPP
