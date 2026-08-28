/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __GTSOPS_OPS_UTIL_H__
#define __GTSOPS_OPS_UTIL_H__

#include <type_traits>
#include <cmath>

namespace ops {
/**
 * if y is 0, return x
 */
template<typename T>
typename std::enable_if<std::is_signed<T>::value, T>::type CeilDiv(T x, T y)
{
    if (y != 0 && x != 0) {
        const T quotient = x / y;
        return (x % y != 0 && ((x ^ y) >= 0)) ? (quotient + 1) : quotient;
    }
    return x;
}

/**
 * if y is 0, return x
 */
template<typename T>
typename std::enable_if<std::is_unsigned<T>::value, T>::type CeilDiv(T x, T y)
{
    if (y != 0 && x != 0) {
        const T quotient = x / y;
        return (x % y != 0) ? (quotient + 1) : quotient;
    }
    return x;
}

/**
 * if y is 0, return x
 */
template<typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type FloorDiv(T x, T y)
{
    return y == 0 ? x : x / y;
}

/**
 * if align is 0, return 0
 */
template<typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type CeilAlign(T x, T align)
{
    return CeilDiv(x, align) * align;
}

/**
 * if align is 0, return 0
 */
template<typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type FloorAlign(T x, T align)
{
    return align == 0 ? 0 : x / align * align;
}

template<typename T>
inline T MathCeil(T num1, T num2)
{
    if (num2 == 0) {
        return num1;
    }
    return (num1 + num2 - 1) / num2;
}

template<typename T>
inline T MathFloor(T num1, T num2)
{
    if (num2 == 0) {
        return num1;
    }
    return num1 / num2;
}

template<typename T>
inline T MathMod(T num1, T num2)
{
    if (num2 == 0) {
        return num1;
    }
    return num1 % num2;
}

template<typename T>
inline T AlignUp(T num1, T num2)
{
    return MathCeil(num1, num2) * num2;
}

template<typename T>
inline T AlignDown(T num1, T num2)
{
    return MathFloor(num1, num2) * num2;
}

template<typename T>
inline bool IsEqual(T l_value, T r_value)
{
    return std::fabs(l_value - r_value) <= std::numeric_limits<T>::epsilon();
}
}
#endif