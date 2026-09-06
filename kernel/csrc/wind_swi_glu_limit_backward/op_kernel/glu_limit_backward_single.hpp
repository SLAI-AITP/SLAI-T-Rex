/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CANN_XX_GLU_LIMIT_BACKWARD_SINGLE_HPP
#define CANN_XX_GLU_LIMIT_BACKWARD_SINGLE_HPP
#include "xxglu_limit_tiling_kernel_single.h"

template<typename ParentClass, typename inType, typename outType>
class GluLimitBackwardSingle : public ParentClass {
public:
    __aicore__ inline GluLimitBackwardSingle() {}
    __aicore__ inline ~GluLimitBackwardSingle() {}
    __aicore__ inline void Init(GM_ADDR grad_gm, GM_ADDR input_gm, GM_ADDR probs_gm, GM_ADDR output_gm,
                                GM_ADDR grad_probs_gm,
                                const XxGluLimitBackwardSingleTilingData* tiling_gm, TPipe& pipe)
    {
        singleTiling.GetTilingAndOffset(tiling_gm, sizeof(inType));
        InitGmBuffer(grad_gm, input_gm, probs_gm, output_gm, grad_probs_gm);
        this->InitUbBuffer(pipe, singleTiling.tileLength);
    }

    __aicore__ inline void Process()
    {
        // grad_probs 切列支持：各列片在 CopyOut 用 SetAtomicAdd 把部分和累加到 gradProbsGm_[row]。
        // GM 预清零由适配层保证(grad_probs 用 at::zeros 分配)，kernel 内无需清零/跨核同步。
        if (singleTiling.is32BAligned == 1) {
            XXGLU_SINGLE_PROCESS(singleTiling);
        } else {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
            XXGLU_SINGLE_PROCESS_NON32BALIGNED(singleTiling);
#endif
        }
    }

protected:
    __aicore__ inline void InitGmBuffer(GM_ADDR grad_gm, GM_ADDR input_gm, GM_ADDR probs_gm, GM_ADDR output_gm,
                                        GM_ADDR grad_probs_gm)
    {
        this->limit = singleTiling.limit; // 透传 clamp 阈值给 Compute 类
        // get start index for current core, core parallel
        this->aGm.SetGlobalBuffer((__gm__ inType*)input_gm, singleTiling.totalBlockLen);
        this->lGm.SetGlobalBuffer((__gm__ inType*)grad_gm, singleTiling.totalBlockLen / 2);

        this->mGm.SetGlobalBuffer((__gm__ outType*)output_gm, singleTiling.totalBlockLen);

        // probs: gradout/out 都是 [rowLen, colLen]，probs 是 [rowLen]。
        // hasProbs=0(probs_gm 为 null) 时跳过，Compute 内 pHasProbs_=0 等价现状。
        this->pHasProbs_ = singleTiling.hasProbs;
        if (this->pHasProbs_ != 0U) {
            this->probsGm_.SetGlobalBuffer((__gm__ float*)probs_gm, singleTiling.rowLen);
        }
        // grad_probs：hasProbs 即算。切列时同行多列片各算部分和，CopyOut 原子累加汇总，故不限制 colTileNum==1。
        this->outputGradProbs_ = (singleTiling.hasProbs != 0U) ? 1U : 0U;
        if (this->outputGradProbs_ != 0U) {
            this->gradProbsGm_.SetGlobalBuffer((__gm__ float*)grad_probs_gm, singleTiling.rowLen);
        }
    }

    __aicore__ inline void CopyIn(XxGluSingleTileOffsetParam &offsetParam, XxGluSinlgeTileCopyParam &copyParam)
    {
        DataCopyParams splitCopyinParams = {copyParam.splitVecCopyParam.blockCount,
                                            copyParam.splitVecCopyParam.blockLen,
                                            copyParam.splitVecCopyParam.stride,
                                            0};
        DataCopyParams indepCopyinParams = {copyParam.indepVecCopyParam.blockCount,
                                            copyParam.indepVecCopyParam.blockLen,
                                            copyParam.indepVecCopyParam.stride,
                                            0};

        // Copy A
        LocalTensor<inType> aLocal = this->inQueueA.template AllocTensor<inType>();
        DataCopy(aLocal, this->aGm[offsetParam.splitVecGmOffset1], splitCopyinParams);
        this->inQueueA.template EnQue(aLocal);
        // Copy L
        LocalTensor<inType> lLocal = this->inQueueL.template AllocTensor<inType>();
        DataCopy(lLocal, this->lGm[offsetParam.indepVecGmoffset], indepCopyinParams);
        this->inQueueL.template EnQue(lLocal);
        // Copy B
        LocalTensor<inType> bLocal = this->inQueueB.template AllocTensor<inType>();
        DataCopy(bLocal, this->aGm[offsetParam.splitVecGmOffset2], splitCopyinParams);
        this->inQueueB.template EnQue(bLocal);

        // probs: 本 tile 的 curRow 个 probs 搬进 UB，并按 token 行广播成
        // [curRow×curCol] 向量(probsVecLocal_)，供 Compute 一次 Mul 乘到 lLocal(=gradout)。全部在 CopyIn 完成，
        // 含 MTE2→S/V 同步，避免 GetValue 读到未搬完的 probs。hasProbs=0 时跳过。
        if (this->pHasProbs_ != 0U) {
            this->pCurFirstRow_ = singleTiling.curFirstRow;
            this->pCurRow_ = singleTiling.curRow;
            this->pCurCol_ = singleTiling.curCol;
            DataCopyParams probsCopyParams = {1, (uint16_t)(singleTiling.curRow * sizeof(float)), 0, 0};
            DataCopyPadParams probsPad = {false, 0, 0, 0};
            ::DataCopyPad(this->probsLocal_, this->probsGm_[singleTiling.curFirstRow], probsCopyParams, probsPad);
            // MTE2→S 同步：等 probs 搬完再 GetValue（EVENT_ID0 与 inQueueA 同 id，但此处串行无重叠，安全）
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            // 逐行广播：第 i 行 curCol 列 = probs[i]
            for (uint32_t i = 0; i < singleTiling.curRow; ++i) {
                float pv = this->probsLocal_.GetValue(i);
                Duplicate<float>(this->probsVecLocal_[i * singleTiling.curCol], pv, singleTiling.curCol);
            }
            // S→V 同步：Duplicate(标量写) 完成后 Compute 才能用 probsVecLocal_ 做 Mul
            pipe_barrier(PIPE_V);
        }
    }

    __aicore__ inline void CopyOut(XxGluSingleTileOffsetParam &offsetParam, XxGluSinlgeTileCopyParam &copyParam)
    {
        DataCopyParams splitCopyoutParams = {copyParam.splitVecCopyParam.blockCount,
                                             copyParam.splitVecCopyParam.blockLen,
                                             0,
                                             copyParam.splitVecCopyParam.stride};

        // deque output tensor from VECOUT queue
        LocalTensor<outType> mLocal = this->outQueueM.template DeQue<outType>();
        // copy progress_th tile from local tensor to global tensor
        DataCopy(this->mGm[offsetParam.splitVecGmOffset1], mLocal, splitCopyoutParams);

        // free output tensor for reuse
        this->outQueueM.template FreeTensor(mLocal);

        // deque output tensor from VECOUT queue
        LocalTensor<outType> nLocal = this->outQueueN.template DeQue<outType>();
        // copy progress_th tile from local tensor to global tensor
        DataCopy(this->mGm[offsetParam.splitVecGmOffset2], nLocal, splitCopyoutParams);

        // free output tensor for reuse
        this->outQueueN.template FreeTensor(nLocal);

        // grad_probs 写出：gpRowLocal_[curRow] 本列片部分和 -> 原子累加到 gradProbsGm_[curFirstRow..]。
        // gpRowLocal_ 是 TBuf，Compute(PIPE_V) 写完需 V→MTE3 同步。
        // SetAtomicNone 前须等 DataCopyPad(MTE3)落盘，否则原子模式提前关→退化成非原子覆盖→多核写同行丢更新。
        if (this->outputGradProbs_ != 0U) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyParams gpCopyParams = {1, (uint16_t)(singleTiling.curRow * sizeof(float)), 0, 0};
            AscendC::SetAtomicAdd<float>();
            DataCopyPad(this->gradProbsGm_[singleTiling.curFirstRow], this->gpRowLocal_, gpCopyParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::SetAtomicNone();
        }
    }

    __aicore__ inline void CopyIn_Non32BAligned(XxGluSingleTileOffsetParam &offsetParam,
                                                XxGluSinlgeTileCopyParam &copyParam)
    {
        DataCopyParams splitCopyinParams = {copyParam.splitVecCopyParam.blockCount,
                                            copyParam.splitVecCopyParam.blockLen,
                                            copyParam.splitVecCopyParam.stride,
                                            0};
        DataCopyParams indepCopyinParams = {copyParam.indepVecCopyParam.blockCount,
                                            copyParam.indepVecCopyParam.blockLen,
                                            copyParam.indepVecCopyParam.stride,
                                            0};
        DataCopyPadParams copyPadParams = {false, 0, 0, 0};
        // Copy A
        LocalTensor<inType> aLocal = this->inQueueA.template AllocTensor<inType>();
        DataCopyPad(aLocal, this->aGm[offsetParam.splitVecGmOffset1], splitCopyinParams, copyPadParams);
        this->inQueueA.template EnQue(aLocal);
        // Copy L
        LocalTensor<inType> lLocal = this->inQueueL.template AllocTensor<inType>();
        DataCopyPad(lLocal, this->lGm[offsetParam.indepVecGmoffset], indepCopyinParams, copyPadParams);
        this->inQueueL.template EnQue(lLocal);
        // Copy B
        LocalTensor<inType> bLocal = this->inQueueB.template AllocTensor<inType>();
        DataCopyPad(bLocal, this->aGm[offsetParam.splitVecGmOffset2], splitCopyinParams, copyPadParams);
        this->inQueueB.template EnQue(bLocal);

        // probs(非32B对齐路径)：UB 每行按 ALIGNUP(curCol) 补齐，
        // 广播 stride 必须用补齐后的列宽，否则与 lLocal 的 padded 布局错位。padding 列乘到 probs 无害(不写出)。
        // 注：MoE shape(3072列,bf16) 恒 32B 对齐，走上面的 CopyIn；本分支仅为通用正确性兜底。
        if (this->pHasProbs_ != 0U) {
            uint32_t alignCol = ALIGNUP(singleTiling.curCol, AscendC::ONE_BLK_SIZE / (uint32_t)sizeof(inType));
            this->pCurFirstRow_ = singleTiling.curFirstRow;
            this->pCurRow_ = singleTiling.curRow;
            this->pCurCol_ = alignCol;
            DataCopyParams probsCopyParams = {1, (uint16_t)(singleTiling.curRow * sizeof(float)), 0, 0};
            DataCopyPadParams probsPad = {false, 0, 0, 0};
            ::DataCopyPad(this->probsLocal_, this->probsGm_[singleTiling.curFirstRow], probsCopyParams, probsPad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            for (uint32_t i = 0; i < singleTiling.curRow; ++i) {
                float pv = this->probsLocal_.GetValue(i);
                Duplicate<float>(this->probsVecLocal_[i * alignCol], pv, alignCol);
            }
            pipe_barrier(PIPE_V);
        }
    }

    __aicore__ inline void CopyOut_Non32BAligned(XxGluSingleTileOffsetParam &offsetParam,
                                                 XxGluSinlgeTileCopyParam &copyParam)
    {
        DataCopyParams splitCopyoutParams = {copyParam.splitVecCopyParam.blockCount,
                                             copyParam.splitVecCopyParam.blockLen,
                                             0,
                                             copyParam.splitVecCopyParam.stride};

        // deque output tensor from VECOUT queue
        LocalTensor<outType> mLocal = this->outQueueM.template DeQue<outType>();
        // copy progress_th tile from local tensor to global tensor
        DataCopyPad(this->mGm[offsetParam.splitVecGmOffset1], mLocal, splitCopyoutParams);
        // free output tensor for reuse
        this->outQueueM.template FreeTensor(mLocal);

        // deque output tensor from VECOUT queue
        LocalTensor<outType> nLocal = this->outQueueN.template DeQue<outType>();
        // copy progress_th tile from local tensor to global tensor
        DataCopyPad(this->mGm[offsetParam.splitVecGmOffset2], nLocal, splitCopyoutParams);
        // free output tensor for reuse
        this->outQueueN.template FreeTensor(nLocal);

        // grad_probs 写出(非32B对齐路径)：与 32B 对齐路径同样处理。SetAtomicNone 前等 DataCopyPad(MTE3)落盘，
        // 否则原子模式提前关→非原子覆盖→多核并发写同行丢更新。
        if (this->outputGradProbs_ != 0U) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyParams gpCopyParams = {1, (uint16_t)(singleTiling.curRow * sizeof(float)), 0, 0};
            AscendC::SetAtomicAdd<float>();
            DataCopyPad(this->gradProbsGm_[singleTiling.curFirstRow], this->gpRowLocal_, gpCopyParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::SetAtomicNone();
        }
    }

private:
    XxgluSingleTilingKernel singleTiling;
};
#endif // CANN_XX_GLU_LIMIT_BACKWARD_SINGLE_HPP
