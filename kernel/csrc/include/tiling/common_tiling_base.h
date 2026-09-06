 /**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GTSOPS_COMMON_TILING_BASE_H
#define GTSOPS_COMMON_TILING_BASE_H
#include "tiling_base.h"

namespace optiling {
class CommonTilingBase : public TilingBaseClass {
constexpr static uint32_t SYS_RESERVE_WS_LEN = 16 * 1024 * 1024; // 16 : 系统预留ws空间, 1024 : mb/kb/byte转换单位
#define SwitchCall(func, ...) context_->func(__VA_ARGS__)
#define SwitchAttrIndex(index) index

public:
    using TilingBaseClass::TilingBaseClass;

    static uint64_t CeilAlign(uint64_t count, uint64_t alignedSize)
    {
        if (alignedSize == 0) {
            return count;
        }
        return (count + alignedSize - 1) / alignedSize * alignedSize;
    }

    static uint64_t CeilDown(uint64_t count, uint64_t alignedSize)
    {
        if (alignedSize == 0) {
            return count;
        }
        return count / alignedSize * alignedSize;
    }

    static uint64_t CeilDiv(const uint64_t n1, const uint64_t n2)
    {
        if (n1 == 0) {
            return 0;
        }
        return (n2 != 0) ? (((n1 - 1) / n2) + 1) : n1;
    }

    static uint64_t Div(const uint64_t n1, const uint64_t n2)
    {
        if (n2 == 0) {
            return 0;
        }
        return n1 / n2;
    }

    static uint64_t Mod(const uint64_t n1, const uint64_t n2)
    {
        if (n2 == 0) {
            return 0;
        }
        return n1 % n2;
    }

    ge::graphStatus GetPlatformInfo() final
    {
        auto platformInfo = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());

        platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, limitUbMemSize_);
        platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::L1, limitL1MemSize_);
        platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, limitL0AMemSize_);
        platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, limitL0BMemSize_);
        platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, limitL0CMemSize_);

        auto limitCoreNum = std::getenv("WIND_LIMIT_CORE_NUM");
        auto limitAicNum = std::getenv("WIND_LIMIT_AIC_NUM");
        auto limitAivNum = std::getenv("WIND_LIMIT_AIV_NUM");
        limitCoreNum_ = (limitCoreNum != nullptr) ? std::strtoul(limitCoreNum, nullptr, 10U) : platformInfo.GetCoreNum();
        limitAicNum_ = (limitCoreNum != nullptr) ? std::strtoul(limitAicNum, nullptr, 10U) : platformInfo.GetCoreNumAic();
        limitAivNum_ = (limitCoreNum != nullptr) ? std::strtoul(limitAivNum, nullptr, 10U) : platformInfo.GetCoreNumAiv();
        if (limitAicNum_ > 0) {
            aivDivAic_ = limitAivNum_ / limitAicNum_;
        }
        socVersion_ = platformInfo.GetSocVersion();
        sysWorkspaceSize_ = platformInfo.GetLibApiWorkSpaceSize();

        if (limitUbMemSize_ <= 0 || limitCoreNum_ <= 0 || limitCoreNum_ > 64U) {
            OP_LOGE(context_->GetNodeName(), "Compile Info is invalid, coreNum:%u, ubSize:%u",
                    limitCoreNum_, limitUbMemSize_);
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    // 3.1 计算Tiling
    virtual ge::graphStatus CalcTiling() = 0;
    // 3.2 校验Tiling结果
    virtual ge::graphStatus CheckTiling() = 0;
    // 6.1 获取用户workspace大小
    virtual size_t GetUserWorkspaceSize() = 0;
    // 7.1 获取计算核数
    [[nodiscard]] virtual uint32_t GetBlockDim() = 0;
    // 7.2 设置TilingData
    virtual ge::graphStatus SaveTilingData(gert::TilingData& tilingData) = 0;

    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() final
    {
        auto result = CalcTiling();
        if (result != ge::GRAPH_SUCCESS) {
            return result;
        }

        return CheckTiling();
    }

public:
    uint64_t GetLimitUbMemSize() const { return limitUbMemSize_; }
    uint32_t GetLimitAivNum() const { return limitAivNum_; }

    [[nodiscard]] size_t GetInputOutputNum() const
    {
        auto nodeInfo = context_->GetComputeNodeInfo();
        if (nodeInfo == nullptr) {
            throw std::invalid_argument("ComputeNodeInfo is nullptr");
        }

        return nodeInfo->GetInputsNum() + nodeInfo->GetOutputsNum();
    }

    [[nodiscard]] ge::graphStatus CheckSameInputs(size_t number) const
    {
        auto inTensor = GetInputTensor(0);
        auto inShape = inTensor->GetOriginShape();
        for (size_t i = 1U; i < number; ++i) {
            auto otherTensor = GetInputTensor(i);
            auto otherShape = otherTensor->GetOriginShape();
            if (inShape != otherShape) {
                throw std::invalid_argument("Input " + std::to_string(i) + " shape is not match");
            }
            if (inTensor->GetDataType() != otherTensor->GetDataType()) {
                throw std::invalid_argument("Input " + std::to_string(i) + " data type is not match");
            }
        }
        return ge::GRAPH_SUCCESS;
    }

    [[nodiscard]] const gert::Tensor* GetInputTensor(size_t index) const
    {
        auto input = SwitchCall(GetInputTensor, index);
        if (input == nullptr) {
            throw std::invalid_argument("Input " + std::to_string(index) + " is nullptr");
        }
        return input;
    }

    [[nodiscard]] const gert::StorageShape* GetInputShape(size_t index) const
    {
        auto input = SwitchCall(GetInputShape, index);
        if (input == nullptr) {
            throw std::invalid_argument("Input " + std::to_string(index) + " is nullptr");
        }
        return input;
    }

    [[nodiscard]] const gert::Tensor* GetOptionalInputTensor(size_t index) const
    {
        return SwitchCall(GetOptionalInputTensor, index);
    }

    const gert::StorageShape *GetOptionalInputShape(size_t index) const
    {
        return SwitchCall(GetOptionalInputShape, index);
    }

    [[nodiscard]] ge::DataType GetInputDataType(size_t index) const
    {
        return GetInputTensor(index)->GetDataType();
    }

    [[nodiscard]] size_t GetInputDataSize(size_t index) const
    {
        return ge::GetSizeByDataType(GetInputDataType(index));
    }

    [[nodiscard]] size_t GetInputShapeSize(size_t index) const
    {
        return GetInputTensor(index)->GetShapeSize();
    }

    [[nodiscard]] ge::Format GetInputFormat(size_t index) const
    {
        return static_cast<ge::Format>(ge::GetPrimaryFormat(GetInputTensor(index)->GetStorageFormat()));
    }

    [[nodiscard]] const gert::StorageShape* GetOutputShape(size_t index) const
    {
        auto shape = SwitchCall(GetOutputShape, index);
        if (shape == nullptr) {
            throw std::invalid_argument("Output shape " + std::to_string(index) + " is nullptr");
        }
        return shape;
    }

    [[nodiscard]] const gert::StorageShape* GetOptionalOutputShape(size_t index) const
    {
        return SwitchCall(GetOutputShape, index);
    }

    [[nodiscard]] ge::Format GetOutputFormat(size_t index) const
    {
        auto outputDesc = context_->GetOutputDesc(index);
        if (outputDesc == nullptr) {
            throw std::invalid_argument("Output desc " + std::to_string(index) + " is nullptr");
        }
        return static_cast<ge::Format>(ge::GetPrimaryFormat(outputDesc->GetStorageFormat()));
    }

    [[nodiscard]] ge::DataType GetOutputDataType(size_t index) const
    {
        auto outputDesc = context_->GetOutputDesc(index);
        if (outputDesc == nullptr) {
            throw std::invalid_argument("Output desc " + std::to_string(index) + " is nullptr");
        }
        return outputDesc->GetDataType();
    }

    [[nodiscard]] const gert::RuntimeAttrs* GetAttrs() const
    {
        auto attrs = context_->GetAttrs();
        if (attrs == nullptr) {
            throw std::invalid_argument("Attrs is nullptr");
        }
        return attrs;
    }

    template <typename T>
    [[nodiscard]] T GetAttr(size_t index) const
    {
        auto newIndex = SwitchAttrIndex(index);
        auto attrs = GetAttrs();
        if (attrs->GetAttrNum() <= newIndex) {
            throw std::invalid_argument("Attr " + std::to_string(newIndex) + " is out of range");
        }

        if constexpr (std::is_same_v<T, std::string>) {
            auto str = attrs->GetStr(newIndex);
            std::string result{str};
            return result;
        } else {
            return *(attrs->GetAttrPointer<T>(newIndex));
        }
    }

    template <typename T>
    [[nodiscard]] T GetAttrWithDefault(size_t index, T defaultValue) const
    {
        auto attrs = GetAttrs();
        auto newIndex = SwitchAttrIndex(index);
        if (attrs->GetAttrNum() <= newIndex) {
            return defaultValue;
        }
        return GetAttr<T>(index);
    }

    [[nodiscard]] size_t GetShapeSize(size_t index) const
    {
        return GetInputTensor(index)->GetShapeSize();
    }

    [[nodiscard]] size_t GetDim(size_t index, size_t dim) const
    {
        auto shape = GetInputTensor(index)->GetStorageShape();
        if (shape.GetDimNum() <= dim) {
            throw std::invalid_argument("Dim " + std::to_string(dim) + " of input " +
                std::to_string(index) + " is out of range");
        }
        return shape.GetDim(dim);
    }

    ge::graphStatus GetShapeAttrsInfo() override { return ge::GRAPH_SUCCESS; }

    bool IsCapable() override { return true; }

    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override { return ge::GRAPH_SUCCESS; }
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() final
    {
        // 设置workspace空间大小
        auto workspaces = context_->GetWorkspaceSizes(1);
        // 如需要使用workspace需要默认设置系统workspace的大小为16MB。
        workspaces[0] = sysWorkspaceSize_ + GetUserWorkspaceSize(); // 预留16M的sysWorkspace

        return ge::GRAPH_SUCCESS;
    }
    // 7、保存Tiling数据
    virtual ge::graphStatus PostTiling() final
    {
        auto blockDim = GetBlockDim();
        if (blockDim == 0) {
            throw std::runtime_error("blockDim is 0");
        }
        context_->SetBlockDim(blockDim);

        auto tilingData = context_->GetRawTilingData();
        if (tilingData == nullptr) {
            return ge::GRAPH_FAILED;
        }

        auto ret = SaveTilingData(*tilingData);
        return ret;
    }

protected:
    // ====================================tiling相关====================================
    bool ParseCustomTiling() {
       auto customTilingChar = std::getenv("WIND_CUSTOM_TILING");
       if (customTilingChar == nullptr) {
           return false;
       }
       std::string customTiling = customTilingChar;
       auto begin = 0;
       auto pos = customTiling.find(',');
       while (pos != std::string::npos) {
           customTiling_.emplace_back(customTiling.substr(begin, pos));
           begin = pos + 1;
           pos = customTiling.find(',', begin);
       }
       customTiling_.emplace_back(customTiling.substr(begin));

       return true;
    }

    template <typename T>
    T GetCustomTiling(size_t index) const {
        if (index >= customTiling_.size()) {
            throw std::invalid_argument("CustomTiling index " + std::to_string(index) + " is out of range");
        }

        if constexpr (std::is_integral_v<T>) {
            return std::stoull(customTiling_[index]);
        } else if constexpr (std::is_floating_point_v<T>) {
            return std::stof(customTiling_[index]);
        }
        throw std::invalid_argument("CustomTiling index " + std::to_string(index) + " is unsupported type");
    }

    std::vector<std::string> customTiling_;

    // ====================================shape相关====================================
    ge::graphStatus CheckMatrixShape(size_t index) const
    {
        auto tensor = GetInputTensor(index);
        auto shape = tensor->GetStorageShape();
        if (tensor->GetStorageFormat() == ge::FORMAT_ND && shape.GetDimNum() != 2U) {
            throw std::invalid_argument("Input ND tensor should be 2D.");
        }

        if (tensor->GetStorageFormat() == ge::FORMAT_FRACTAL_NZ && shape.GetDimNum() != 4U) {
            throw std::invalid_argument("Input NZ tensor should be 4D.");
        }
        return ge::GRAPH_SUCCESS;
    }

    // 返回：M|N, K
    std::tuple<int64_t, int64_t> GetMNK(size_t index, bool isLeftMatrix, bool isTranspose = false) const
    {
        auto tensor = GetInputTensor(index);
        auto shape = tensor->GetStorageShape();
        if (shape.GetDimNum() < 2U) {
            throw std::runtime_error("Input " + std::to_string(index) + " is not a matrix");
        }

        bool newTranspose = isLeftMatrix ? isTranspose : !isTranspose;
        switch (tensor->GetStorageFormat()) {
            case ge::Format::FORMAT_ND:
                // 左矩阵——非转置：M*K, 转置：K*M
                // 右矩阵——非转置：K*N, 转置：N*K
                // 结果——非转置：M*N，转置:N*M（不常用）
                return {newTranspose ? shape.GetDim(1) : shape.GetDim(0), newTranspose ? shape.GetDim(0) : shape.GetDim(1)};
            case ge::Format::FORMAT_FRACTAL_NZ:
                // ===海思格式 fp16下m0=k0=n0=16
                //  左矩阵——非转置：k1, m1, m0, k0, 转置：m1, k1, k0, m0
                //  右矩阵——非转置：n1, k1, k0, n0, 转置：k1, n1, n0, k0
                //  结果——非转置：n1, m1, m0, n0，转置：（不常用）
                return {newTranspose ? (shape.GetDim(1U) * shape.GetDim(2U)) : (shape.GetDim(0U) * shape.GetDim(3U)),
                        newTranspose ? (shape.GetDim(0U) * shape.GetDim(3U)) : (shape.GetDim(1U) * shape.GetDim(2U))};

                // ===计算格式 fp16下m0=k0=n0=16
                //  左矩阵——非转置：1, m1, k1*k0, m0, 转置：1, k1, m1*m0, k0
                //  右矩阵——非转置：1, k1, n1*n0, k0, 转置：1, n1, k1*k0, n0
                // return {newTranspose ? shape.GetDim(2U) : (shape.GetDim(1U) * shape.GetDim(3U)),
                //         newTranspose ? (shape.GetDim(1U) * shape.GetDim(3U)) : shape.GetDim(2U)};
            default:
                throw std::runtime_error("Unsupported format " + std::to_string(tensor->GetStorageFormat()));
        }
    }

protected:
    uint64_t limitUbMemSize_ = 0;
    uint64_t limitL1MemSize_ = 0;
    uint64_t limitL0AMemSize_ = 0;
    uint64_t limitL0BMemSize_ = 0;
    uint64_t limitL0CMemSize_ = 0;
    uint32_t limitCoreNum_ = 0;
    uint32_t limitAicNum_ = 0;
    uint32_t limitAivNum_ = 0;
    uint32_t aivDivAic_ = 2;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    size_t sysWorkspaceSize_ = 0;
};
}
#endif // GTSOPS_COMMON_TILING_BASE_H
