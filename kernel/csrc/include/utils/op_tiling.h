/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_TILING_UTIL_H
#define OP_TILING_UTIL_H
#include <iostream>
#include "utils/op_math.h"
#include <type_traits>
using namespace std;

constexpr static uint32_t SYS_RESERVE_WS_LEN = 16 * 1024 * 1024; // 16 : 系统预留ws空间, 1024 : mb/kb/byte转换单位

constexpr float eps_float = 1e-8;
#define EquFloat(a, b) (fabs((a) - (b)) < (eps_float))

constexpr static uint16_t DEFAULT_MIN_BLOCK_SIZE = 32; // 最小的数据块长度，32Bytes

// BSH,SBH,BSND,BNSD
#define INPUT_LAYOUT_BSH 0
#define INPUT_LAYOUT_SBH 1
#define INPUT_LAYOUT_BSND 2
#define INPUT_LAYOUT_BNSD 3
#define INPUT_LAYOUT_TND 4

#define ATTEN_MASK_LAYOUT_SS 0
#define ATTEN_MASK_LAYOUT_B1SS 1
#define ATTEN_MASK_LAYOUT_BNSS 2
#define ATTEN_MASK_LAYOUT_TS 3

#define FA_S_MAX_BATCH 3840
#define FA_S_MAX_SEQ_LEN 32768
#define FA_S_MAX_HEAD_NUM 256
#define FA_S_MAX_HEAD_DIM 256
#define FA_S_MIN_ALIGN_LEN 16 // headdim要求必须16对齐
#define FA_S_SM_MAX_HEADDIM 8 // softmax max/sum输出的headdim，固定为8
#define FA_S_SPARSE_MAX_TOKENS 2147483647

#define FA_S_MAX_Q_BASE_LEN 256 // 切换切分的最大长度，单位:element
#define FA_S_MAX_KV_BASE_LEN 128 // 切换切分的最大长度，单位:element
#define FA_S_MAX_KV_BASE_LEN_PA 256 // PA 切换切分的最大长度，单位:element

#define FA_S_MAX_BATCH_PERCORE 30

#define ASSIGN_CORE_TYPE_BYQBLK 0
#define ASSIGN_CORE_TYPE_BYHEAD 1
#define ASSIGN_CORE_TYPE_BYHEAD_SKIP 2 // 按照head分核，但是每个core处理的head是跳跃的
#define ASSIGN_CORE_TYPE_BYBATCH 3 // 按照batch分核
#define ASSIGN_CORE_TYPE_BYGHEAD 4 // 按照global head分核
#define ASSIGN_CORE_TYPE_BYSBLK 5 // 按照sb分块分核

// Nz格式类型---ND最后2维变为Nz之后的格式
#define NZ_FMT_N1MN0 0 // atb输入Nz格式
#define NZ_FMT_N1M1M0N0 1 // cann输入Nz格式
constexpr static uint8_t BOOL_MASK = 0;
constexpr static uint8_t F16_MASK = 1; // FP16,BF16
constexpr static uint16_t B32_DATA_NUM_PER_BLOCK = 8;
constexpr static uint16_t BLOCK_CUBE_LEN = 16; // cube核上一个分型的长度 单位:element
constexpr static uint16_t FLOAT_BLOCK_LENS = 8; // 仅float使用，单位:element
constexpr static uint16_t FLOAT_REPEAT_LENS = 64; // 仅float使用，单位:element
constexpr static uint16_t MAX_REPEAT_STRIDE = 255;
constexpr static uint16_t REPEATE_LEN_BYTES = 256; // 一个repeat的长度，单位:bytes
constexpr static uint64_t UINT16_MAX_VALUE = 65535;

constexpr static uint16_t PINGPONGBUFFERNUM = 2; // inner-group aic/aiv sync
constexpr static uint16_t MIN_QKV_BASELEN = 16; // 16 : 最小分块单位 k/q/v最小分块长度16，单位：element
constexpr static uint16_t MIN_KV_BASELEN = 128; // // 16 : 最小分块单位 k/v最小分块长度64，单位：element
constexpr static uint16_t MAX_BLOCKNUM_PER_KVBASE = 2; // PA时 一次kvbase基块包含的block数量
constexpr static uint16_t MAX_SYNC_KVHEAD_NUM = 8;
constexpr static uint16_t  WIND_PA_MAX_CALC_PARAM_NUM = 70;
constexpr static uint16_t  WIND_PA_MAX_CALC_UNIT_NUM = 5;
constexpr static uint16_t  WIND_MAX_PP_COMB_BLK_NUM = 8;

constexpr static uint64_t MAX_KV_L1_BUFFER_NUM = 20; // L1上k/v上最大的buffer数量
constexpr static uint64_t MAX_PA_KV_L1_BUFFER_NUM = 20; // L1上k/v上最大的buffer数量

// 增量阶段的masktype
enum IncreMaskType {
    UNDEFINED,           // 默认值，全0的mask
    MASK_TYPE_NORM,      // 倒三角mask
    MASK_TYPE_ALIBI,     // alibi mask
    MASK_TYPE_SPEC,       // 并行解码mask
    MASK_TYPE_MASK_FREE  // 并行解码mask free
};

struct WindPlatformStruct {
    uint64_t limitUbMemSize = 0;
    uint64_t limitL1MemSize = 0;
    uint64_t limitL0AMemSize = 0;
    uint64_t limitL0BMemSize = 0;
    uint64_t limitL0CMemSize = 0;
    uint16_t limitCoreNum = 0;
    uint16_t limitAicNum = 0;
    uint16_t limitAivNum = 0;
    uint16_t aivDivAic = 2;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;

    inline bool Is310P()
    {
        return socVersion == platform_ascendc::SocVersion::ASCEND310P;
    }

    inline void Print()
    {
        std::cout << "limitUbMemSize: " << limitUbMemSize << std::endl;
        std::cout << "limitL1MemSize: " << limitL1MemSize << std::endl;
        std::cout << "limitL0AMemSize: " << limitL0AMemSize << std::endl;
        std::cout << "limitL0BMemSize: " << limitL0BMemSize << std::endl;
        std::cout << "limitL0CMemSize: " << limitL0CMemSize << std::endl;
        std::cout << "limitCoreNum: " << limitCoreNum << std::endl;
        std::cout << "limitAicNum: " << limitAicNum << std::endl;
        std::cout << "limitAivNum: " << limitAivNum << std::endl;
        std::cout << "aivDivAic: " << aivDivAic << std::endl;
    }
};

struct OptTilingParam {
    bool isValid;
    uint16_t qBaseLen;
    uint16_t kvBaseLen;

    uint16_t usedBlockNum;
    uint32_t totalQBlkNum;
    uint32_t qBlkNumPerCore; // 1个核能分到的qBlk数量（不能整除时向上取整），越少越好

    // 内存使用情况
    uint64_t ubUsedSize;
    uint64_t l1UsedSize;
    uint64_t l0aUsedSize;
    uint64_t l0bUsedSize;
    uint64_t l0cUsedSize;
    // 内存使用率
    float ubUsageRatio;
    float l1UsageRatio;
    float l0aUsageRatio;
    float l0bUsageRatio;
    float l0cUsageRatio;

    inline void Print()
    {
        std::cout << "=== OptTilingParam ===" << std::endl;
        std::cout << "isValid: " << isValid << std::endl;
        std::cout << "qBaseLen: " << qBaseLen << std::endl;
        std::cout << "kvBaseLen: " << kvBaseLen << std::endl;
        std::cout << "totalQBlkNum: " << totalQBlkNum << std::endl;
        std::cout << "usedBlockNum: " << usedBlockNum << std::endl;
        std::cout << "qBlkNumPerCore: " << qBlkNumPerCore << std::endl;
        std::cout << "ubUsedSize: " << ubUsedSize << std::endl;
        std::cout << "l1UsedSize: " << l1UsedSize << std::endl;
        std::cout << "l0aUsedSize: " << l0aUsedSize << std::endl;
        std::cout << "l0bUsedSize: " << l0bUsedSize << std::endl;
        std::cout << "l0cUsedSize: " << l0cUsedSize << std::endl;
        std::cout << "ubUsageRatio: " << ubUsageRatio << std::endl;
        std::cout << "l1UsageRatio: " << l1UsageRatio << std::endl;
        std::cout << "l0aUsageRatio: " << l0aUsageRatio << std::endl;
        std::cout << "l0bUsageRatio: " << l0bUsageRatio << std::endl;
        std::cout << "l0cUsageRatio: " << l0cUsageRatio << std::endl;
    }
};

struct BatchInfo {
    uint16_t qBlkNum; // 该batch的qBlk分块数量总数
    uint16_t aclSeqLen; // 该batch实际的seq长度
    uint16_t seqQblkTailLen; // 尾块的seq的长度
    uint16_t qBlkNumPerHead; // 每个head有几个head
    uint16_t qBlkNumPerBatch; // 每个batch有几个head
    uint16_t startQBlkIdx; // 每个batch的起始全局QBlkIdx

    inline void Print(uint16_t batchIdx)
    {
        std::cout << "BatchInfo: batchIdx:" << batchIdx << " qBlkNum:" << qBlkNum << " aclSeqLen:" << aclSeqLen << " seqQblkTailLen:" << seqQblkTailLen <<
            " qBlkNumPerHead:" << qBlkNumPerHead << " qBlkNumPerBatch:" << qBlkNumPerBatch << " startQBlkIdx:" << startQBlkIdx << std::endl;
    }
};

// 分配到每个core的分块信息，以batch维度管理
struct AssignBatchInfo {
    uint16_t batchIdx;
    uint16_t qBlkNum; // 该batch分到改core的qBlk分块数量
    uint16_t startqBlkIdx; // 该batch分到改core的qBlk起始索引,本batch内的索引编号
};

struct AssignBatchInCore {
    uint16_t batchNum; // 本core分配的的query分块所涉及的所有batch数量
    // 本core分陪的batch信息，实际有效数量与batchNumInCore一直
    AssignBatchInfo batchInfos[FA_S_MAX_BATCH_PERCORE];

    inline void Print(uint32_t coreIdx)
    {
        for (int32_t batchLoop = 0; batchLoop < batchNum; batchLoop++) {
            std::cout << coreIdx << " | " << batchNum << " | "
                      << batchInfos[batchLoop].batchIdx << " | " << batchInfos[batchLoop].qBlkNum
                      << " | " << batchInfos[batchLoop].startqBlkIdx << std::endl;
        }
    }
};

struct QBLKClusterStru {
    uint16_t qBlkNumPerCluster; // 每个Cluster包含多少个qBlkNum
    uint16_t qBlkNumTailPerCluster; // 如果不能均匀分割，每个Cluster包含多少个qBlkNum
    uint64_t clusterNumPerGroup; // 每个group可以分割多少个qBlkCluster
    uint64_t clusterNum; // 所有batch可以分割多少个qBlkCluster

    void Print()
    {
        std::cout << "--QBLKCluster  qBlkNumPerCluster: " << qBlkNumPerCluster
                  << " qBlkNumTailPerCluster: " << qBlkNumTailPerCluster
                  << " clusterNumPerGroup: " << clusterNumPerGroup
                  << " clusterNum: " << clusterNum << std::endl;
    }
};

struct GcdClusterItemStru {
    // 实体 index范围是左闭右开区间 [startIdx, endIdx)
    uint16_t itemNum; // 实体数量， 资源可能是batch/group/head/core
    uint16_t startIdx; // 开始实体index， 实体可能是batch/group/head/core
    uint16_t endIdx; // 结束实体index
};

// 按照最大公约数的分组信息
struct GcdClusterStru {
    uint16_t clusterNum;
    // 资源index范围是闭区间 [startResIdx, endResIdx]
    GcdClusterItemStru *resItems; // 资源，可能是batch/group/head
    GcdClusterItemStru *coreItems; // aic核信息

    void Print(std::string label)
    {
        std::cout << "---" << label << " GcdCluster info---" << std::endl;
        std::cout << "clusterNum: " << clusterNum << std::endl;
        std::cout << "resNum | resStartIdx | resEndIdx | coreNum | coreStartIdx | coreEndIdx" << std::endl;

        for (int32_t i = 0; i < clusterNum; i++) {
            std::cout << resItems[i].itemNum << " | " << resItems[i].startIdx << " | "<< resItems[i].endIdx << " | "
                      << coreItems[i].itemNum << " | " << coreItems[i].startIdx << " | "<< coreItems[i].endIdx << std::endl;
        }
    }
};

// 每个核分配到的qBlk信息， 取值范围左闭右开 [startBlkIdx, endBlkIdx)
struct CoreAssignBlkStru {
    uint32_t startBlkIdx;
    uint32_t endBlkIdx;
    uint64_t calcVolume; // 计算量，稀疏计算时存在
};

// 使用最大公约数的方式分组， 获取itemNum和coreNum的最大公约数。如果最大公约数为1，按照不均匀的分组
/*****************************************************************
 * 辗转相除算法
 * 例1: 求13和20的最大公约数
 * a  |  b  |  a mod b
 * 20    13    6
 * 13    6     1
 * 6     1     0
 * 13和20的最大公约数是1，但是按照不均匀分组，返回6+1,同时设置为不能均分
 *
 * 例2: 求14和20的最大公约数
 * a  |  b  |  a mod b
 * 20    14    2
 * 14    2     0
 * 14和20的最大公约数是2
 *****************************************************************/
inline uint16_t GetClusterNumByGcd(uint16_t itemNum, uint16_t coreNum, bool& isEquDiv)
{
    if (itemNum == 0 || coreNum == 0) {
        return 0;
    }

    isEquDiv = true;
    if (itemNum == coreNum) {
        return itemNum;
    }
    if (itemNum == 1 || coreNum == 1) {
        return 1;
    }

    uint16_t a = std::max(itemNum, coreNum);
    uint16_t b = std::min(itemNum, coreNum);
    uint16_t r = OpMod(a, b);

    while (r != 0) {
        a = b;
        b = r;
        r = OpMod(a, b);
    }

    isEquDiv = b > 1; // 最大公约数gcd=b，若大于1说明存在其他最大公约数，可以按照gcd均分
    uint16_t clusterNum = isEquDiv ? b : (a + 1); // 如果没有除1之外的最大公约数，按照不均匀方式分组
    return clusterNum > coreNum ? 1 : clusterNum; // 如果超出corenum，按照1组来处理
}

// 将totalItemNum尽量平均分为clusterNum组， idxOffset是totalItemNum个示例的起始index编号，itemList是输出
inline void SplitItemForCluster(uint16_t totalItemNum, uint16_t clusterNum, uint16_t idxOffset,
                                GcdClusterItemStru *itemList)
{
    if (clusterNum == 0) {
        return;
    }
    uint16_t laterClusterNum = clusterNum - totalItemNum % clusterNum;
    uint16_t formerClusterNum = clusterNum - laterClusterNum;
    uint16_t itemNum_formerCluster = OpDivCeil(totalItemNum, clusterNum, 0);
    uint16_t itemNum_laterCluster = totalItemNum / clusterNum;
    for (int32_t i = 0; i < formerClusterNum; i++) {
        itemList[i].itemNum = itemNum_formerCluster;
        itemList[i].startIdx = idxOffset + i * itemNum_formerCluster;
        itemList[i].endIdx = itemList[i].startIdx + itemList[i].itemNum;
    }
    for (int32_t i = 0; i < laterClusterNum; i++) {
        uint32_t itemPos = formerClusterNum + i;
        itemList[itemPos].itemNum = itemNum_laterCluster;
        itemList[itemPos].startIdx =
                idxOffset + formerClusterNum * itemNum_formerCluster + i * itemNum_laterCluster;
        itemList[itemPos].endIdx = itemList[itemPos].startIdx + itemList[itemPos].itemNum;
    }
}

// 将coreNum个分块均分到coreNum上，结果里的索引都是相对的
inline void EquAssignBlkToCore(uint32_t blkNum, uint16_t coreNum, bool hasTail, CoreAssignBlkStru* assignResults)
{
    if (coreNum == 0) {
        return;
    }

    // 所有qBlk按照顺序分配给所有核， 以aic为单位进行分核 (startGlobalQBlkIdx, endGlobalQBlkIdx]
    uint16_t laterCoreNum;
    uint16_t formerCoreNum;
    uint32_t maxBlkNum_formerCore;
    uint32_t maxBlkNum_laterCore;
    if (hasTail) {
        // 有尾块时，formerCore和laterCore反过来用， 从后往前分配
        formerCoreNum = coreNum - blkNum % coreNum;
        laterCoreNum = coreNum - formerCoreNum;
        maxBlkNum_formerCore = blkNum / coreNum;
        maxBlkNum_laterCore = OpDivCeil(blkNum, coreNum, 0);
    } else {
        laterCoreNum = coreNum - blkNum % coreNum;
        formerCoreNum = coreNum - laterCoreNum;
        maxBlkNum_formerCore = OpDivCeil(blkNum, coreNum, 0);
        maxBlkNum_laterCore = blkNum / coreNum;
    }

    // 记录每个核的信息
    for (int32_t coreLoop = 0; coreLoop < coreNum; coreLoop++) {
        // 本core处理的qBlk范围 [startGlobalBlkIdx, endGlobalBlkIdx), 左闭右开
        uint32_t startGlobalBlkIdx = 0; // 所有batch统一编号的全局qBlkIdx
        uint32_t endGlobalBlkIdx = 0; // 所有batch统一编号的全局qBlkIdx
        if (coreLoop < formerCoreNum) {
            startGlobalBlkIdx = coreLoop * maxBlkNum_formerCore;
            endGlobalBlkIdx = startGlobalBlkIdx + maxBlkNum_formerCore;
        } else {
            startGlobalBlkIdx = formerCoreNum * maxBlkNum_formerCore + (coreLoop - formerCoreNum) * maxBlkNum_laterCore;
            endGlobalBlkIdx = startGlobalBlkIdx + maxBlkNum_laterCore;
        }

        // 将分配到qBlk信息保存到对应core的数据中
        assignResults[coreLoop].startBlkIdx = startGlobalBlkIdx;
        assignResults[coreLoop].endBlkIdx = endGlobalBlkIdx;
    }
}

// 返回的元素个数
inline uint64_t GetReduceXxNdTempBuffLen_float(uint64_t rowLen, uint64_t colLen)
{
    if (colLen <= 8U) {
        return rowLen * 8U;
    } else if ((colLen == 64U) || (colLen == 128U) || (colLen == 256U) || (colLen == 512U) || (colLen == 1024U)) {
        return DIVCEIL(rowLen * colLen, 64U) * 8U;
    } else {
        uint64_t tailColLen = MOD(colLen, 64U);
        uint64_t tailReduceBlkNum = (tailColLen !=0 && tailColLen != 8U) ? 1 : 0;
        uint64_t reduceBlkNum = DIVCEIL(colLen, 64U) + tailReduceBlkNum;
        return reduceBlkNum * rowLen * 8U;
    }
}

inline uint64_t GetReduceXxNdTempBuffLen_half(uint64_t rowLen, uint64_t colLen)
{
    if (colLen <= 16U) {
        return rowLen * 16U;
    } else if (colLen == 32U) {
        return DIVCEIL(rowLen, 8U) * 128U;
    } else if (colLen == 128U) {
        return DIVCEIL(rowLen, 2U) * 128U;
    } else {
        return DIVCEIL(colLen, 128U) * rowLen * 16U;
    }
}

/*
 * 计算Wind::ReduceXx_Nd接口需要临时buffer大小（元素），对应reduceLocal
 * isPerfFirst为false时：
 *   reduceLocal是计算过程使用的临时缓存，最小缓存大小 rowLen * repeatLens（float为64，half为128）
 *   float类型colLen最大支持到255*64=2040； half类型的colLen最大支持到255*128=32640
 * isPerfFirst为true时：
 *   colLen可以比较大，没有限制
 *   reduceLocal需要的内存规则
 *   ┌───────┬────────────────────────────────┬────────────────────────────┬────────────────────────────┐
 *   │ 类型  │          colLen 条件           │   reduceLocal 最小元素数   │   reduceLocal 最小字节数   │
 *   ├───────┼────────────────────────────────┼────────────────────────────┼────────────────────────────┤
 *   │ float │ colLen == 8                    │ 0                          │ 0                          │
 *   ├───────┼────────────────────────────────┼────────────────────────────┼────────────────────────────┤
 *   │ float │ colLen ∈ {64,128,256,512,1024} │ ceil(rowLen×colLen/64)×8   │ ceil(rowLen×colLen/64)×32  │
 *   ├───────┼────────────────────────────────┼────────────────────────────┼────────────────────────────┤
 *   │ float │ 其他                           │ reduceBlkNum×rowLen×8      │ reduceBlkNum×rowLen×32     │
 *   ├───────┼────────────────────────────────┼────────────────────────────┼────────────────────────────┤
 *   │ half  │ colLen == 16                   │ 0                          │ 0                          │
 *   ├───────┼────────────────────────────────┼────────────────────────────┼────────────────────────────┤
 *   │ half  │ colLen == 32                   │ ceil(rowLen/8)×128         │ ceil(rowLen/8)×256         │
 *   ├───────┼────────────────────────────────┼────────────────────────────┼────────────────────────────┤
 *   │ half  │ colLen == 128                  │ ceil(rowLen/2)×128         │ ceil(rowLen/2)×256         │
 *   ├───────┼────────────────────────────────┼────────────────────────────┼────────────────────────────┤
 *   │ half  │ 其他                           │ ceil(colLen/128)×rowLen×16 │ ceil(colLen/128)×rowLen×32 │
 *   └───────┴────────────────────────────────┴────────────────────────────┴────────────────────────────┘
 *   其中 reduceBlkNum(浮点) = colLen/64 + isTail(tailColLen)
 *   tailColLen = colLen % 64
 *   isTail(t) = (t != 0 && t != 8) ? 1 : 0
 * */
template <typename T, bool isPerfFirst = true>
inline uint64_t GetReduceXxNdTempBuffLen(uint64_t rowLen, uint64_t colLen)
{
    if constexpr (isPerfFirst) {
        if constexpr (std::is_same<T, float>::value) {
            return GetReduceXxNdTempBuffLen_float(rowLen, colLen);
        } else {
            return GetReduceXxNdTempBuffLen_half(rowLen, colLen);
        }
    } else {
        return rowLen * DIVCEIL(256U, sizeof(T));
    }
}

#endif  // OP_TILING_UTIL_H

