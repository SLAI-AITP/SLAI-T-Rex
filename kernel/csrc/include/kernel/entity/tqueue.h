/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_TQUEUE_H
#define OP_TQUEUE_H
#include "kernel_operator.h"

using namespace AscendC;

template<typename Type, uint8_t bufferNum, HardEvent enQueEvt, HardEvent freeBufEvt>
struct TQueue {
    __aicore__ inline TQueue(TEventID evtId) : evtId_(evtId)
    {
        for (int i = 0; i < bufferNum; i++) {
            enQueEvtId_[i] = INVALID_TEVENTID;
            freeBufEvtId_[i] = INVALID_TEVENTID;
        }
    }

    __aicore__ inline ~TQueue()
    {
        for (int i = 0; i < bufferNum; i++) {
            if (freeBufEvtId_[i] != INVALID_TEVENTID) {
                WaitFlag<freeBufEvt>(freeBufEvtId_[i]);
            }
        }
    }

    __aicore__ inline void Init(TPipe& pipe, uint64_t tileLength)
    {
        pipe.InitBuffer(buf_, tileLength * sizeof(Type) * bufferNum);
        tileLength_ = tileLength;
        for (int i = 0; i < bufferNum; i++) {
            tensor_[i] = buf_.GetWithOffset<Type>(tileLength_, tileLength_ * sizeof(Type) * i);
        }
    }

    template<typename T>
    __aicore__ inline LocalTensor<T>& AllocTensor()
    {
        idx_ = ((idx_ == bufferNum - 1) ? 0 : idx_ + 1);
        if (freeBufEvtId_[idx_] != INVALID_TEVENTID) {
            WaitFlag<freeBufEvt>(freeBufEvtId_[idx_]);
            freeBufEvtId_[idx_] = INVALID_TEVENTID;
        }
        return tensor_[idx_];
    }

    template<typename T>
    __aicore__ inline void FreeTensor(LocalTensor<T>& tensor)
    {
        freeBufEvtId_[idx_] = evtId_ + idx_;
        SetFlag<freeBufEvt>(freeBufEvtId_[idx_]);
    }

    template<typename T>
    __aicore__ inline void EnQue(const LocalTensor<T>& tensor)
    {
        enQueEvtId_[idx_] = evtId_ + idx_;
        SetFlag<enQueEvt>(enQueEvtId_[idx_]);
    }

    template<typename T>
    __aicore__ inline LocalTensor<T>& DeQue()
    {
        if (enQueEvtId_[idx_] != INVALID_TEVENTID) {
            WaitFlag<enQueEvt>(enQueEvtId_[idx_]);
            enQueEvtId_[idx_] = INVALID_TEVENTID;
        }
        return tensor_[idx_];
    }

    LocalTensor<Type> tensor_[bufferNum];
    TBuf<TPosition::VECCALC> buf_;
    uint64_t tileLength_;
    TEventID enQueEvtId_[bufferNum];
    TEventID freeBufEvtId_[bufferNum];
    TEventID evtId_;

    uint8_t idx_ = bufferNum - 1;
};

template<typename Type, uint8_t bufferNum>
using InQue = TQueue<Type, bufferNum, HardEvent::MTE2_V, HardEvent::V_MTE2>;

template<typename Type, uint8_t bufferNum>
using OutQue = TQueue<Type, bufferNum, HardEvent::V_MTE3, HardEvent::MTE3_V>;

#endif // OP_TQUEUE_H
