# SLAI-T-Rex加速库构建&使用指南

## (推荐) 方式1：直接使用镜像
镜像已完成加速特性相关补丁的安装和版本构建

## 方式2：基于源码构建
### step1 拉取MindSpeed和MindSpeed-LLM开源源码
```sh
mkdir opensource
cd opensource
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout -b develop v26.1.0_core_r0.12.1
cd -

git clone https://gitcode.com/Ascend/MindSpeed-LLM.git
cd MindSpeed-LLM
git checkout -b develop v26.1.0
cd -
```
### step2 apply问题修复/算子接入/优化特性补丁
执行如下命令
```sh
bash <代码根目录>/script/patch.sh <opensource路径>
```
### step3 算子包构建&安装
```sh
# 以A3硬件为例，若为A2--soc ascend910b
bash <代码根目录>/script/build.sh all --ops "wind_*" --soc "ascend910_93"
```
构建成功后会算子包会自动安装到代码根目录的vendors目录下

### step4 训练脚本source加速包
在训练脚本中添加如下命令
```sh
source <代码根目录>/script/set_env.sh
```
### step5 验证执行单算子
```sh
cd <代码根目录>/patch/gts/ops
python gts_custom_op.py
```