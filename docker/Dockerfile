# syntax=docker/dockerfile:1.4

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PYTHON_VERSION=3.10.18
ARG CANN_VERSION=9.1.0
ARG TORCH_VERSION=2.7.1
ARG TORCH_NPU_VERSION=2.7.1.post8
ARG TORCHVISION_VERSION=0.22.1
ARG TORCHAUDIO_VERSION=2.7.1
ARG TRITON_ASCEND_VERSION=3.2.2
ARG NUMPY_VERSION=1.26.4
ARG TRANSFORMERS_VERSION=5.2.0
ARG PYARROW_VERSION=24.0.0
ARG RAY_VERSION=2.57.0
ARG MINDSPEED_REF=v26.1.0_core_r0.12.1
ARG MINDSPEED_LLM_REF=v26.1.0
ARG MEGATRON_REF=core_v0.12.1
ARG ENABLE_TCMALLOC=1
ARG GTS_SOC=ascend910_93
ARG GTS_OPS=wind_*
ARG OPS_TRANSFORMER_OPS
ARG OPS_TRANSFORMER_REF=unknown
ARG SLAI_TREX_REF=unknown

LABEL org.opencontainers.image.title="SLAI T-Rex Ascend training image" \
      org.opencontainers.image.description="MindSpeed-LLM 26.1.0 with SLAI T-Rex GTS and ops-transformer" \
      org.opencontainers.image.source="https://gitcode.com/SLAI-AI-training-platform/SLAI-T-Rex" \
      io.mindspeed.ref="${MINDSPEED_REF}" \
      io.megatron.ref="${MEGATRON_REF}" \
      io.mindspeed-llm.ref="${MINDSPEED_LLM_REF}" \
      io.slai-trex.ref="${SLAI_TREX_REF}" \
      io.ops-transformer.ref="${OPS_TRANSFORMER_REF}"

USER root
SHELL ["/bin/bash", "-c"]

ENV OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    BLIS_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN dnf install -y \
        autoconf \
        automake \
        bzip2-devel \
        ca-certificates \
        cmake \
        curl \
        expat-devel \
        findutils \
        gcc \
        gcc-c++ \
        gcc-gfortran \
        gdbm-devel \
        git \
        gzip \
        iproute \
        lapack-devel \
        libffi-devel \
        libtool \
        libuuid-devel \
        make \
        net-tools \
        ncurses-devel \
        openssl-devel \
        patch \
        pciutils \
        procps-ng \
        readline-devel \
        sqlite-devel \
        tar \
        unzip \
        wget \
        which \
        xz \
        xz-devel \
        zlib-devel \
    && dnf clean all \
    && rm -rf /var/cache/dnf

RUN --mount=type=bind,source=packages,target=/mnt/packages \
    mkdir -p /tmp/python-build \
    && tar -xf "/mnt/packages/Python-${PYTHON_VERSION}.tar.xz" -C /tmp/python-build \
    && cd "/tmp/python-build/Python-${PYTHON_VERSION}" \
    && ./configure \
        --prefix="/usr/local/python${PYTHON_VERSION}" \
        --enable-shared \
        --with-ensurepip=install \
    && make -j"$(nproc)" \
    && make install \
    && rm -rf /tmp/python-build

ENV PYTHON_HOME=/usr/local/python3.10.18 \
    PATH=/usr/local/python3.10.18/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/python3.10.18/lib

RUN python3 --version \
    && test "$(python3 -c 'import platform; print(platform.python_version())')" = "${PYTHON_VERSION}"

RUN --mount=type=bind,source=packages,target=/mnt/packages \
    bash "/mnt/packages/Ascend-cann-toolkit_${CANN_VERSION}_linux-aarch64.run" --quiet --install \
    && bash "/mnt/packages/Ascend-cann-A3-ops_${CANN_VERSION}_linux-aarch64.run" --quiet --install \
    && source /usr/local/Ascend/ascend-toolkit/set_env.sh \
    && bash "/mnt/packages/Ascend-cann-nnal_${CANN_VERSION}_linux-aarch64.run" --quiet --install

ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest \
    ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp \
    ATB_HOME_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1 \
    PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe: \
    LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/python3.10.18/lib \
    PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/bin:/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin:/usr/local/python3.10.18/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN python3 -m pip install --no-cache-dir --upgrade \
        pip==25.1 \
        setuptools==78.1.1 \
        wheel==0.45.1 \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN python3 -m pip install --no-cache-dir \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && python3 -m pip install --no-cache-dir \
        "torch-npu==${TORCH_NPU_VERSION}" \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN python3 -m pip install --no-cache-dir \
        "triton-ascend==${TRITON_ASCEND_VERSION}" \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
        --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/

COPY requirements.txt /opt/mindspeed/requirements.txt
COPY validate_image.py /opt/mindspeed/validate_image.py

RUN python3 -m pip install --no-cache-dir \
        --requirement /opt/mindspeed/requirements.txt \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN set -euo pipefail; \
    mkdir -p /workspace /opt/mindspeed; \
    fetch_source() { \
        local name="$1" url="$2" ref="$3" target="$4"; \
        git init --quiet "$target"; \
        git -C "$target" remote add origin "$url"; \
        for attempt in 1 2 3 4 5; do \
            if git -C "$target" fetch --quiet --depth 1 origin "$ref"; then \
                break; \
            fi; \
            if [ "$attempt" -eq 5 ]; then \
                echo "Unable to fetch ${name} ref ${ref}" >&2; \
                exit 1; \
            fi; \
            sleep 5; \
        done; \
        git -C "$target" checkout --quiet --detach FETCH_HEAD; \
        printf '%s %s %s\n' "$name" "$(git -C "$target" rev-parse HEAD)" "$ref" \
            >> /opt/mindspeed/training-source-manifest.txt; \
    }; \
    fetch_source MindSpeed https://gitcode.com/Ascend/MindSpeed.git "$MINDSPEED_REF" /workspace/MindSpeed; \
    fetch_source Megatron-LM https://github.com/NVIDIA/Megatron-LM.git "$MEGATRON_REF" /workspace/Megatron-LM; \
    fetch_source MindSpeed-LLM https://gitcode.com/Ascend/MindSpeed-LLM.git "$MINDSPEED_LLM_REF" /workspace/MindSpeed-LLM; \
    python3 -m pip install --no-cache-dir --no-deps --editable /workspace/MindSpeed; \
    ln -s /workspace/MindSpeed/mindspeed /workspace/MindSpeed-LLM/mindspeed; \
    ln -s /workspace/Megatron-LM/megatron /workspace/MindSpeed-LLM/megatron; \
    make -C /workspace/Megatron-LM/megatron/core/datasets -j1; \
    mkdir -p /workspace/MindSpeed-LLM/logs; \
    python3 -m pip freeze --all | LC_ALL=C sort > /opt/mindspeed/python-packages.lock.txt

RUN if ! id ma-user >/dev/null 2>&1; then \
        useradd --uid 1000 --create-home --shell /bin/bash ma-user; \
    fi \
    && mkdir -p /home/ma-user/modelarts/user-job-dir \
    && chown -R 1000:1000 /home/ma-user

COPY <<'EOF' /usr/local/bin/mindspeed-entrypoint.sh
#!/bin/bash
set -e
if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
    source /usr/local/Ascend/cann/set_env.sh
elif [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
unset TORCH_DEVICE_BACKEND_AUTOLOAD
cd /workspace/MindSpeed-LLM
exec "$@"
EOF

ENV MINDSPEED_HOME=/workspace/MindSpeed \
    MEGATRON_HOME=/workspace/Megatron-LM \
    MINDSPEED_LLM_HOME=/workspace/MindSpeed-LLM \
    PYTHONPATH=/workspace/MindSpeed-LLM:/workspace/MindSpeed:/workspace/Megatron-LM:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe:

RUN chmod 0755 /usr/local/bin/mindspeed-entrypoint.sh \
    && env TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
        python3 -u /opt/mindspeed/validate_image.py

ENV GTS_ROOT=/workspace/SLAI-T-Rex/kernel \
    GTS_INSTALL_PATH=/workspace/SLAI-T-Rex/kernel \
    TORCH_SITE=/usr/local/python3.10.18/lib/python3.10/site-packages \
    PYTHONDONTWRITEBYTECODE=1

ENV PYTHONPATH=${GTS_ROOT}/op-speed:${PYTHONPATH} \
    LD_LIBRARY_PATH=${TORCH_SITE}/torch_npu/lib:${TORCH_SITE}/torch/lib:${GTS_ROOT}/vendors/gtsops/op_api/lib:${ASCEND_OPP_PATH}/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH} \
    ASCEND_CUSTOM_OPP_PATH=${ASCEND_OPP_PATH}/vendors/custom_transformer:${GTS_ROOT}/vendors/gtsops

COPY --from=gts_kernel / ${GTS_ROOT}/

RUN cd "${GTS_ROOT}" \
    && bash ./script/patch.sh /workspace \
    && test -f /workspace/MindSpeed/.patch.enable \
    && test -f /workspace/MindSpeed-LLM/.patch.enable

RUN set -euo pipefail; \
    source /usr/local/Ascend/cann/set_env.sh; \
    ln -sfn "$(command -v python3)" /usr/local/bin/python; \
    cd "${GTS_ROOT}/csrc"; \
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; \
    export BUILD_OPS_LIST="${GTS_OPS}"; \
    export BUILD_SOC="${GTS_SOC}"; \
    export GTS_BUILD_TARGET=build_out; \
    bash ./build.sh; \
    mapfile -t gts_runs < <(find build_out -maxdepth 1 -type f -name 'gtsops_*.run' | sort); \
    test "${#gts_runs[@]}" -eq 1; \
    bash "${gts_runs[0]}" --install-path="${GTS_ROOT}"; \
    test -f "${GTS_ROOT}/vendors/gtsops/op_api/lib/libcust_opapi.so"; \
    rm -rf "${ASCEND_OPP_PATH}/vendors/gtsops"; \
    if [ -f "${ASCEND_OPP_PATH}/vendors/config.ini" ]; then \
        sed -i 's/^load_priority=gtsops$/load_priority=/' \
            "${ASCEND_OPP_PATH}/vendors/config.ini"; \
    fi

RUN set -euo pipefail; \
    source /usr/local/Ascend/cann/set_env.sh; \
    ln -sfn "$(command -v python3)" /usr/local/bin/python; \
    cd "${GTS_ROOT}/op-speed"; \
    export PYTHON_BIN="$(command -v python3)"; \
    export PYTHON_INCLUDE_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_path("include") or "")')"; \
    export SITE_PACKAGES_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib") or "")')"; \
    export PKG_TYPE=CANN-S; \
    python3 setup.py build_ext --inplace; \
    test -f "${GTS_ROOT}/op-speed/gtsops.so"; \
    ! ldd "${GTS_ROOT}/op-speed/gtsops.so" | grep -q 'not found'; \
    python3 -c 'import gtsops; print("gtsops ready:", gtsops.__file__)'

COPY --from=gperftools / /opt/src/gperftools-2.18.1/

RUN set -euo pipefail; \
    if [ "${ENABLE_TCMALLOC}" = "1" ]; then \
        cd /opt/src/gperftools-2.18.1; \
        if [ ! -x ./configure ]; then \
            test -x ./autogen.sh; \
            ./autogen.sh; \
        fi; \
        ./configure \
            --prefix="${GTS_ROOT}/third_party/malloc/tcmalloc" \
            --enable-minimal; \
        make -j"$(nproc)"; \
        make install; \
        test -f "${GTS_ROOT}/third_party/malloc/tcmalloc/lib/libtcmalloc_minimal.so.4"; \
    fi; \
    rm -rf /opt/src/gperftools-2.18.1

COPY --from=ops_transformer / /opt/src/ops-transformer/

RUN set -euo pipefail; \
    source /usr/local/Ascend/cann/set_env.sh; \
    cd /opt/src/ops-transformer; \
    rm -rf build build_out; \
    bash ./build.sh \
        --pkg \
        --experimental \
        --soc="${GTS_SOC}" \
        --ops="${OPS_TRANSFORMER_OPS}"; \
    mapfile -t ops_runs < <(find build build_out -maxdepth 2 -type f \
        -name 'cann-ops-transformer-custom_linux-aarch64.run' 2>/dev/null | sort -u); \
    test "${#ops_runs[@]}" -eq 1; \
    bash "${ops_runs[0]}" --quiet; \
    python3 -m pip install \
        --no-build-isolation \
        --no-deps \
        --upgrade \
        --target "${ASCEND_HOME_PATH}/python/site-packages" \
        /opt/src/ops-transformer/torch_extension; \
    test -f "${ASCEND_OPP_PATH}/vendors/custom_transformer/bin/set_env.bash"; \
    rm -rf build build_out

WORKDIR /workspace/MindSpeed-LLM
ENTRYPOINT ["/usr/local/bin/mindspeed-entrypoint.sh"]
CMD ["tail", "-f", "/dev/null"]
