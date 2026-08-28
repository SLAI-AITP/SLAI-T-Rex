/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GTSOPS_OP_MATH_H
#define GTSOPS_OP_MATH_H

// align num to multiples of rnd, round up
#define ALIGNUP(num, rnd) (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)))
// align num to multiples of rnd, round down
#define ALIGNDOWN(num, rnd) ((((rnd) == 0) || ((num) < (rnd))) ? 0 : ((num) / (rnd) * (rnd)))
// div and Round Up
#define DIVCEIL(num, div) (((div) == 0) ? 0 : (((num) + (div) - 1) / (div)))
#define DIVFLOOR(num, div) (((div) == 0) ? 0 : ((num) / (div)))
#define MOD(num, div) (((div) == 0) ? 0 : ((num) - (num) / (div) * (div)))
#define MIN(_a, _b) ((_a) > (_b) ? (_b) : (_a))
#define MAX(_a, _b) ((_a) > (_b) ? (_a) : (_b))

template<typename Divisor, typename Dividends, typename Out = int64_t>
#ifdef __aicore__
__aicore__
#endif
inline Out OpDiv(Divisor divisor, Dividends dividends, Out defaultVal = 0)
{
    static_assert(std::is_integral<Divisor>::value, "Divisor expects the type to be integer type");
    static_assert(std::is_integral<Dividends>::value, "Dividends expects the type to be integer type");
    if (dividends == 0) {
        return defaultVal;
    }

    if (divisor < dividends) {
        return 0;
    }

    return divisor / dividends;
}

#ifdef __aicore__
__aicore__
#endif
inline int64_t Ceil(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

#ifdef __aicore__
__aicore__
#endif
inline int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    const int64_t BLOCK_BYTES = 32;
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
}

#ifdef __aicore__
__aicore__
#endif
inline int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    const int64_t BLOCK_BYTES = 32;
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}

template <typename T>
#ifdef __aicore__
__aicore__
#endif
inline T OpMin(T a, T b)
{
    return a > b ? b : a;
}

template <typename T>
#ifdef __aicore__
__aicore__
#endif
inline T OpMax(T a, T b)
{
    return a < b ? b : a;
}

template<typename Divisor, typename Dividends, typename Out = int64_t>
#ifdef __aicore__
__aicore__
#endif
inline Out OpMod(Divisor divisor, Dividends dividends, Out defaultVal = 0)
{
    static_assert(std::is_integral<Divisor>::value, "Divisor expects the type to be integer type");
    static_assert(std::is_integral<Dividends>::value, "Dividends expects the type to be integer type");
    if (dividends == 0) {
        return defaultVal;
    }

    if (divisor < dividends) {
        return divisor;
    }
    return divisor - (divisor / dividends) * dividends;
}

// div结果向上取整
template<typename Divisor, typename Dividends, typename Out>
#ifdef __aicore__
__aicore__
#endif
inline Out OpDivCeil(Divisor divisor, Dividends dividends, Out defaultVal = 0)
{
    static_assert(std::is_integral<Divisor>::value, "Divisor expects the type to be integer type");
    static_assert(std::is_integral<Dividends>::value, "Dividends expects the type to be integer type");
    if (dividends == 0) {
        return defaultVal;
    }
    if (divisor <= dividends) {
        return 1;
    }
    return (divisor + dividends - 1) / dividends;
}

template<typename Divisor, typename Dividends>
#ifdef __aicore__
__aicore__
#endif
inline Divisor DivCeil(Divisor a, Dividends b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

// GE数据类型转换接口
/****************
at:: 枚举值	    对应的 C++数据类型	 数值
at::kFloat	    float	         6
at::kHalf	    at::Half	     5
at::kBFloat16	at::BFloat16	 15
*****************/
#ifndef __aicore__
static constexpr int64_t AT_DTYPE_KHALF = 5;
static constexpr int64_t AT_DTYPE_KFLOAT = 6;
static constexpr int64_t AT_DTYPE_KBFLOAT16 = 15;

inline ge::DataType ConvertAtDtypeToGeDtype(int64_t out_dtype, ge::DataType default_dtype)
{
    if (out_dtype == AT_DTYPE_KFLOAT) {
        return ge::DataType::DT_FLOAT;
    } else if (out_dtype == AT_DTYPE_KHALF) {
        return ge::DataType::DT_FLOAT16;
    } else if (out_dtype == AT_DTYPE_KBFLOAT16) {
        return ge::DataType::DT_BF16;
    }
    return default_dtype;
}
#endif // __aicore__
#endif // GTSOPS_OP_MATH_H
