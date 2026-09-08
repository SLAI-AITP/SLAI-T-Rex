#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
VERSION_FILE="${SCRIPT_DIR}/versions.env"
PACKAGE_DIR="${SCRIPT_DIR}/packages"

# shellcheck disable=SC1090
source "${VERSION_FILE}"

OUTPUT_IMAGE="${OUTPUT_IMAGE:-slai-trex:ascend-cann910-a3}"
BUILD_CACHE_DIR="${BUILD_CACHE_DIR:-${SCRIPT_DIR}/.build-cache}"
OPS_SOURCE="${OPS_SOURCE:-${BUILD_CACHE_DIR}/ops-transformer}"
GPERF_SOURCE="${GPERF_SOURCE:-${BUILD_CACHE_DIR}/gperftools-2.18.1}"
ENABLE_TCMALLOC="${ENABLE_TCMALLOC:-1}"
NO_CACHE="${NO_CACHE:-0}"
BUILDKIT_PROGRESS="${BUILDKIT_PROGRESS:-plain}"
OS_BASE_IMAGE="mindspeed-openeuler:22.03-lts-sp4-${OPEN_EULER_SNAPSHOT//-/}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

for command_name in curl docker git sha256sum tar uname; do
    command -v "${command_name}" >/dev/null 2>&1 \
        || die "required command not found: ${command_name}"
done

case "$(uname -m)" in
    aarch64|arm64) ;;
    *) die "the Ascend training image must be built on Linux AArch64" ;;
esac

case "${ENABLE_TCMALLOC}" in
    0|1) ;;
    *) die "ENABLE_TCMALLOC must be 0 or 1" ;;
esac

docker info >/dev/null
test -f "${SCRIPT_DIR}/Dockerfile" || die "Dockerfile not found"
test -f "${SCRIPT_DIR}/requirements.txt" \
    || die "requirements.txt not found"
test -f "${SCRIPT_DIR}/validate_image.py" \
    || die "validate_image.py not found"
test -f "${REPO_ROOT}/kernel/script/patch.sh" \
    || die "GTS sources not found below ${REPO_ROOT}/kernel"

download_and_verify() {
    local url="$1"
    local target="$2"
    local expected_sha256="$3"
    local partial="${target}.part"

    if [ -f "${target}" ] \
        && printf '%s  %s\n' "${expected_sha256}" "${target}" \
            | sha256sum --check --status; then
        return
    fi

    if [ -f "${target}" ]; then
        mv "${target}" "${target}.invalid.$(date +%s)"
    fi

    curl -L --fail --retry 5 --retry-delay 5 --continue-at - \
        --output "${partial}" "${url}"
    printf '%s  %s\n' "${expected_sha256}" "${partial}" \
        | sha256sum --check --status
    mv "${partial}" "${target}"
}

fetch_source() {
    local name="$1"
    local url="$2"
    local ref="$3"
    local target="$4"
    local required_file="$5"

    if [ -e "${target}" ]; then
        test -f "${target}/${required_file}" \
            || die "existing ${name} source is incomplete: ${target}"

        if [ -d "${target}/.git" ]; then
            local actual expected
            actual=$(git -C "${target}" rev-parse HEAD)
            expected=$(git -C "${target}" rev-parse "${ref}^{commit}" 2>/dev/null || true)
            if [ -n "${expected}" ] && [ "${actual}" != "${expected}" ]; then
                die "${name} source is at ${actual}, expected ${expected} (${ref})"
            fi
            if [[ "${ref}" =~ ^[0-9a-f]{40}$ ]] && [ "${actual}" != "${ref}" ]; then
                die "${name} source is at ${actual}, expected ${ref}"
            fi
        fi

        echo "Using existing ${name} source: ${target}"
        return
    fi

    mkdir -p "$(dirname "${target}")" "${target}"
    git -C "${target}" init --quiet
    git -C "${target}" remote add origin "${url}"

    local attempt
    for attempt in 1 2 3 4 5; do
        if git -C "${target}" fetch --quiet --depth 1 origin "${ref}"; then
            git -C "${target}" checkout --quiet --detach FETCH_HEAD
            break
        fi
        if [ "${attempt}" -eq 5 ]; then
            die "failed to fetch ${name} ref ${ref} from ${url}"
        fi
        sleep 5
    done

    test -f "${target}/${required_file}" \
        || die "fetched ${name} source is incomplete: ${target}"
}

mkdir -p "${PACKAGE_DIR}" "${BUILD_CACHE_DIR}"

OPEN_EULER_FILE=openEuler-docker.aarch64.tar.xz
PYTHON_FILE="Python-${PYTHON_VERSION}.tar.xz"
TOOLKIT_FILE="Ascend-cann-toolkit_${CANN_VERSION}_linux-aarch64.run"
A3_OPS_FILE="Ascend-cann-A3-ops_${CANN_VERSION}_linux-aarch64.run"
NNAL_FILE="Ascend-cann-nnal_${CANN_VERSION}_linux-aarch64.run"

OPEN_EULER_SHA256=80881e6d60b8ebbc24941a9692188471cb14c56096ac510c2e720a3d46ed479e
PYTHON_SHA256=ae665bc678abd9ab6a6e1573d2481625a53719bc517e9a634ed2b9fefae3817f
TOOLKIT_SHA256=ccfa4249422eb60355e425e89a8bddeaf27fd91575ea805334e7e74d7518abd2
A3_OPS_SHA256=bb97028372b181d872dfee1201e83fc3dfaa0728efb51a4afdc8f8b8751311f3
NNAL_SHA256=62a79f474280eecdc5a584d78036a2074ad740369150e9403f260655307d86e1

OPEN_EULER_URL="https://repo.openeuler.org/openEuler-22.03-LTS-SP4/docker_img/update/${OPEN_EULER_SNAPSHOT}/aarch64/${OPEN_EULER_FILE}"
PYTHON_URL="https://repo.huaweicloud.com/python/${PYTHON_VERSION}/${PYTHON_FILE}"
CANN_BASE_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20${CANN_VERSION}"

download_and_verify "${OPEN_EULER_URL}" "${PACKAGE_DIR}/${OPEN_EULER_FILE}" "${OPEN_EULER_SHA256}"
download_and_verify "${PYTHON_URL}" "${PACKAGE_DIR}/${PYTHON_FILE}" "${PYTHON_SHA256}"
download_and_verify "${CANN_BASE_URL}/${TOOLKIT_FILE}" "${PACKAGE_DIR}/${TOOLKIT_FILE}" "${TOOLKIT_SHA256}"
download_and_verify "${CANN_BASE_URL}/${A3_OPS_FILE}" "${PACKAGE_DIR}/${A3_OPS_FILE}" "${A3_OPS_SHA256}"
download_and_verify "${CANN_BASE_URL}/${NNAL_FILE}" "${PACKAGE_DIR}/${NNAL_FILE}" "${NNAL_SHA256}"

docker load -i "${PACKAGE_DIR}/${OPEN_EULER_FILE}"
docker tag openeuler-22.03-lts-sp4:latest "${OS_BASE_IMAGE}"

fetch_source \
    ops-transformer \
    "${OPS_TRANSFORMER_URL}" \
    "${OPS_TRANSFORMER_REF}" \
    "${OPS_SOURCE}" \
    build.sh

if [ "${ENABLE_TCMALLOC}" = "1" ]; then
    fetch_source \
        gperftools \
        "${GPERFTOOLS_URL}" \
        "${GPERFTOOLS_REF}" \
        "${GPERF_SOURCE}" \
        CMakeLists.txt
    GPERF_CONTEXT="${GPERF_SOURCE}"
else
    GPERF_CONTEXT="${BUILD_CACHE_DIR}/empty-gperftools"
    mkdir -p "${GPERF_CONTEXT}"
fi

SLAI_TREX_REF=$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)

export DOCKER_BUILDKIT=1
build_args=(
    --network host
    --progress "${BUILDKIT_PROGRESS}"
    --build-context "gts_kernel=${REPO_ROOT}/kernel"
    --build-context "ops_transformer=${OPS_SOURCE}"
    --build-context "gperftools=${GPERF_CONTEXT}"
    --build-arg "BASE_IMAGE=${OS_BASE_IMAGE}"
    --build-arg "PYTHON_VERSION=${PYTHON_VERSION}"
    --build-arg "CANN_VERSION=${CANN_VERSION}"
    --build-arg "TORCH_VERSION=${TORCH_VERSION}"
    --build-arg "TORCH_NPU_VERSION=${TORCH_NPU_VERSION}"
    --build-arg "TORCHVISION_VERSION=${TORCHVISION_VERSION}"
    --build-arg "TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION}"
    --build-arg "TRITON_ASCEND_VERSION=${TRITON_ASCEND_VERSION}"
    --build-arg "NUMPY_VERSION=${NUMPY_VERSION}"
    --build-arg "TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION}"
    --build-arg "PYARROW_VERSION=${PYARROW_VERSION}"
    --build-arg "RAY_VERSION=${RAY_VERSION}"
    --build-arg "MINDSPEED_REF=${MINDSPEED_REF}"
    --build-arg "MINDSPEED_LLM_REF=${MINDSPEED_LLM_REF}"
    --build-arg "MEGATRON_REF=${MEGATRON_REF}"
    --build-arg "ENABLE_TCMALLOC=${ENABLE_TCMALLOC}"
    --build-arg "GTS_SOC=${GTS_SOC}"
    --build-arg "GTS_OPS=${GTS_OPS}"
    --build-arg "OPS_TRANSFORMER_OPS=${OPS_TRANSFORMER_OPS}"
    --build-arg "OPS_TRANSFORMER_REF=${OPS_TRANSFORMER_REF}"
    --build-arg "SLAI_TREX_REF=${SLAI_TREX_REF}"
    --file "${SCRIPT_DIR}/Dockerfile"
    --tag "${OUTPUT_IMAGE}"
)
if [ "${NO_CACHE}" = "1" ]; then
    build_args+=(--no-cache)
fi

docker build "${build_args[@]}" "${SCRIPT_DIR}"

echo
echo "Image build completed:"
echo "  ${OUTPUT_IMAGE}"
