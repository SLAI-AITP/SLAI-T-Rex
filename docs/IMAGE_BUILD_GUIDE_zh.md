# SLAI T-Rex Ascend 训练镜像搭建指南

本构建用于 DeepSeek-V4 Flash/Pro CPT 与 SFT，目标设备为 Ascend A3
`ascend910_93`。主要版本如下：

- openEuler 22.03 LTS SP4、CANN 9.1.0、Python 3.10.18；
- PyTorch 2.7.1、torch-npu 2.7.1.post8；
- MindSpeed `v26.1.0_core_r0.12.1`；
- MindSpeed-LLM `v26.1.0`；
- Megatron-LM `core_v0.12.1`；
- ops-transformer 9.1.0。

训练数据、Tokenizer、模型权重和集群配置不写入镜像。

## 1. 一键构建

在 Linux AArch64 构建机上安装 Docker，并确保 Docker BuildKit 与公开软件源可访问。
然后在仓库根目录执行：

```bash
bash docker/build_image.sh
```

默认输出镜像为：

```text
slai-trex:ascend-cann910-a3
```

整个过程只调用一次 `docker build`，不会先生成 MindSpeed-LLM 中间镜像。入口脚本只在
Docker 构建前完成安装包下载、SHA256 校验、公开源码准备和 openEuler 基础镜像导入。

## 2. 可选设置

自定义最终镜像名：

```bash
export OUTPUT_IMAGE="slai-trex:ascend-cann910-a3"
bash docker/build_image.sh
```

不编译 tcmalloc：

```bash
export ENABLE_TCMALLOC="0"
bash docker/build_image.sh
```

忽略 Docker 构建缓存：

```bash
export NO_CACHE="1"
bash docker/build_image.sh
```

无法访问 GitCode 或 GitHub 时，可以指向已经准备好的源码目录：

```bash
export OPS_SOURCE="/path/to/ops-transformer"
export GPERF_SOURCE="/path/to/gperftools-2.18.1"
bash docker/build_image.sh
```

`OPS_SOURCE` 必须对应 `versions.env` 固定的 ops-transformer commit；
`GPERF_SOURCE` 必须对应 gperftools 2.18.1。

## 3. 单次 Docker 构建包含的内容

[Dockerfile](../docker/Dockerfile) 在一次 `docker build` 中按以下顺序完成全部工作。

### 3.1 安装基础软件栈

构建首先安装 openEuler 系统依赖，并从源码安装 Python 3.10.18。随后安装 CANN 9.1.0 Toolkit、A3 kernels、NNAL，以及版本锁定的 PyTorch、torch-npu 和 Python 依赖。Python 安装在 `/usr/local/python3.10.18`，CANN 安装在 `/usr/local/Ascend`，后续框架和算子均基于这套环境编译，避免运行时出现 Python ABI、PyTorch ABI 或 CANN 版本不一致。

### 3.2 获取并安装训练框架

Dockerfile 按 [versions.env](../docker/versions.env) 中声明的 tag 获取 MindSpeed、MindSpeed-LLM 和 Megatron-LM，并分别放到 `/workspace/MindSpeed`、`/workspace/MindSpeed-LLM` 和 `/workspace/Megatron-LM`。MindSpeed 以 editable 方式安装，MindSpeed-LLM 通过固定目录及源码链接使用对应的 MindSpeed 和 Megatron-LM，确保训练实际加载的是镜像内这三份匹配的源码，而不是其他挂载目录中的副本。

### 3.3 编译 Megatron 数据集扩展

框架源码准备完成后，Dockerfile 会编译 `megatron.core.datasets.helpers_cpp`。该 C++ 扩展负责构建和读取 Megatron 数据集索引；提前编译进镜像后，训练节点启动时无需再次现场编译，也不会因为缺少 `helpers_cpp` 在创建 dataloader 阶段退出。

### 3.4 应用 GTS 框架补丁

构建上下文中的 `kernel` 目录会复制到 `/workspace/SLAI-T-Rex/kernel`，然后执行 `kernel/script/patch.sh /workspace`。该脚本把 `kernel/patch/gts` 接入 `/workspace/MindSpeed/mindspeed/gts`，并依次将 `kernel/patch/MindSpeed` 和 `kernel/patch/MindSpeed-LLM` 下的补丁应用到镜像内框架。补丁负责增加 GTS 算子的 Python 接口和训练链路替换逻辑；若补丁与所选框架 tag 不匹配，Docker 构建会在此步骤直接失败。

### 3.5 编译并安装 GTS 算子

补丁完成后，构建会针对 `ascend910_93` 编译 `wind_*` 算子。生成的 NPU kernel、算子原型和 op_api 安装到 `/workspace/SLAI-T-Rex/kernel/vendors/gtsops`，Python/C++ 桥接扩展生成在 `/workspace/SLAI-T-Rex/kernel/op-speed/gtsops.so`。镜像同时配置对应的 `PYTHONPATH`、`LD_LIBRARY_PATH` 和 `ASCEND_CUSTOM_OPP_PATH`；训练时再由 `set_env.sh` 决定具体启用哪些 GTS 替换项。

### 3.6 构建 tcmalloc

默认从仓库内置的 `kernel/third_party/gperftools-2.18.1` 源码构建 minimal tcmalloc，不依赖 Docker 构建期间访问 GitHub。库文件安装到 `/workspace/SLAI-T-Rex/kernel/third_party/malloc/tcmalloc`。它随镜像发布，但不会由 Dockerfile 全局强制预加载；训练脚本 source `kernel/script/set_env.sh` 并启用 tcmalloc 后，才会通过 `LD_PRELOAD` 使用它管理主机侧内存。

### 3.7 编译并安装 ops-transformer 注意力算子

最后，Dockerfile 使用镜像内的 CANN 和 Python 环境编译以下 ops-transformer 算子：

```text
sparse_flash_mla
sparse_flash_mla_metadata
sparse_flash_mla_grad
lightning_indexer
sparse_lightning_indexer_grad_kl_loss
```

其中 `sparse_flash_mla`、`sparse_flash_mla_metadata` 和 `sparse_flash_mla_grad` 分别覆盖稀疏 MLA 的前向、元数据生成和反向；`lightning_indexer` 及其 KL-loss 反向算子用于 DSA indexer 链路。构建产物安装到 CANN 的 `custom_transformer` vendor 目录，相应 Python 接口安装到 CANN Python 目录，使 MindSpeed-LLM 可以通过 `cann_ops_transformer` 直接调用。

以上步骤属于同一次镜像构建。任一步骤返回非零状态，`docker build` 都会失败，不会生成最终镜像标签。

## 4. 配套文件

```text
docker/
├── Dockerfile
├── build_image.sh
├── versions.env
├── requirements.txt
└── validate_image.py
```

- `build_image.sh`：唯一构建入口；
- `Dockerfile`：完整镜像构建过程；
- `versions.env`：软件版本、源码 ref、SoC 和算子列表；
- `requirements.txt`：固定 Python 依赖；
- `validate_image.py`：Docker 构建期间的基础一致性检查。

下载的安装包保存在 `docker/packages/`，公开源码缓存在
`docker/.build-cache/`。这两个目录均不会提交到 Git。
