#!/usr/bin/env python3
"""Validate the pinned framework and Python environment during image build."""

import importlib.metadata as metadata
import importlib.util
import pathlib
import threading

from packaging.version import Version


EXPECTED_VERSIONS = {
    "numpy": "1.26.4",
    "pyarrow": "24.0.0",
    "ray": "2.57.0",
    "torch": "2.7.1",
    "torch-npu": "2.7.1.post8",
    "torchaudio": "2.7.1",
    "torchvision": "0.22.1",
    "transformers": "5.2.0",
    "triton-ascend": "3.2.2",
}
LOCAL_VERSION_PACKAGES = {"torch", "torchaudio", "torchvision"}


def check_versions():
    print("===== package versions =====", flush=True)
    for name, expected in EXPECTED_VERSIONS.items():
        actual = metadata.version(name)
        print(f"{name}: actual={actual} expected={expected}", flush=True)
        if name in LOCAL_VERSION_PACKAGES:
            assert Version(actual).base_version == expected, (name, actual, expected)
        else:
            assert actual == expected, (name, actual, expected)


def check_threads():
    print("===== thread creation =====", flush=True)
    thread = threading.Thread(target=lambda: print("thread ok", flush=True))
    thread.start()
    thread.join()


def check_sources():
    print("===== training sources =====", flush=True)
    expected_roots = {
        "mindspeed": pathlib.Path("/workspace/MindSpeed"),
        "mindspeed_llm": pathlib.Path("/workspace/MindSpeed-LLM"),
        "megatron.core": pathlib.Path("/workspace/Megatron-LM"),
    }
    for name, root in expected_roots.items():
        spec = importlib.util.find_spec(name)
        assert spec is not None and spec.origin is not None, name
        resolved = pathlib.Path(spec.origin).resolve()
        print(f"{name}: {resolved}", flush=True)
        assert root in resolved.parents, (name, resolved, root)

    manifest = pathlib.Path("/opt/mindspeed/training-source-manifest.txt")
    print(manifest.read_text(), end="", flush=True)


def check_dataset_helper():
    print("===== Megatron dataset helper =====", flush=True)
    import megatron.core.datasets.helpers_cpp as helpers_cpp

    print(helpers_cpp.__file__, flush=True)


def check_deepseek4_script():
    print("===== DeepSeek4 script =====", flush=True)
    script_path = pathlib.Path(
        "/workspace/MindSpeed-LLM/examples/mcore/deepseek4_flash/"
        "pretrain_deepseek4_flash_4k_A3_ptd.sh"
    )
    script = script_path.read_text()
    assert "--position-embedding-type deepseek4" in script
    print(script_path, flush=True)


def check_torch_cpu():
    print("===== PyTorch CPU matrix =====", flush=True)
    import torch

    x = torch.ones((8, 8))
    checksum = (x @ x).sum().item()
    print(f"checksum={checksum}", flush=True)
    assert checksum == 512.0


def main():
    check_versions()
    check_threads()
    check_sources()
    check_dataset_helper()
    check_deepseek4_script()
    check_torch_cpu()
    print("MindSpeed-LLM 26.1.0 candidate build checks passed", flush=True)


if __name__ == "__main__":
    main()
