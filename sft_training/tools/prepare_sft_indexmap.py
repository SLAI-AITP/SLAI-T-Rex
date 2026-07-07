#!/usr/bin/env python3
"""Prepare shuffle index maps for MindSpeed packed SFT datasets.

MindSpeed's packed instruction dataset looks for files named like:

    {data_prefix}_{split_name}_indexmap_{num_samples}ns_{seed}s_shuffle_decoder_packed_idx.npy

This helper pre-builds that file before distributed SFT starts, so non-zero
nodes can wait for a ready marker instead of racing to build the same index.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-prefix", help="Prefix passed to MindSpeed --data-path.")
    parser.add_argument("--data-dir", help="Dataset directory; used with --prefix-name if --data-prefix is omitted.")
    parser.add_argument("--prefix-name", default="alpaca", help="Prefix basename under --data-dir. Default: alpaca.")
    parser.add_argument("--split-name", default="train", help="Dataset split name used in the indexmap filename.")
    parser.add_argument("--num-samples", type=int, help="Number of samples required by training.")
    parser.add_argument("--train-iters", type=int, help="Training iterations; used with --global-batch-size.")
    parser.add_argument("--global-batch-size", type=int, help="Global batch size; used with --train-iters.")
    parser.add_argument("--doc-count", type=int, help="Document count. Skips automatic inference when set.")
    parser.add_argument("--source-jsonl", help="JSONL file used only to infer document count.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--wait", action="store_true", help="Wait for an existing valid indexmap.")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--padded-samples", action="store_true")
    return parser.parse_args()


def resolve_data_prefix(args: argparse.Namespace) -> Path:
    if args.data_prefix:
        return Path(args.data_prefix)
    if args.data_dir:
        return Path(args.data_dir) / args.prefix_name
    raise ValueError("either --data-prefix or --data-dir must be provided")


def resolve_num_samples(args: argparse.Namespace) -> int:
    if args.num_samples is not None:
        return args.num_samples
    if args.train_iters is None or args.global_batch_size is None:
        raise ValueError("set --num-samples or both --train-iters and --global-batch-size")
    return args.train_iters * args.global_batch_size


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def infer_doc_count_from_stats(data_prefix: Path) -> int | None:
    stats_path = data_prefix.parent / "stats.json"
    if not stats_path.exists():
        return None

    with stats_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    stats = payload.get("stats", payload)
    written = stats.get("written")
    if written is not None and int(written) > 0:
        return int(written)
    return None


def infer_doc_count_from_indexed_dataset(data_prefix: Path) -> int | None:
    try:
        from megatron.core.datasets.indexed_dataset import IndexedDataset
    except Exception:
        return None

    for idx_path in sorted(data_prefix.parent.glob(f"{data_prefix.name}_packed_*_document.idx")):
        indexed_prefix = idx_path.with_suffix("")
        try:
            return len(IndexedDataset(str(indexed_prefix)))
        except Exception:
            continue
    return None


def infer_doc_count_from_jsonl(data_prefix: Path, source_jsonl: str | None) -> int | None:
    if source_jsonl:
        return count_jsonl(Path(source_jsonl))

    candidates = []
    candidates.extend(data_prefix.parent.glob(f"{data_prefix.name}.jsonl"))
    candidates.extend(data_prefix.parent.glob(f"*_{data_prefix.name}.jsonl"))
    candidates.extend(data_prefix.parent.glob("*.jsonl"))

    unique = []
    seen = set()
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique.append(path)

    non_empty = [(path, count_jsonl(path)) for path in unique if path.is_file()]
    non_empty = [(path, count) for path, count in non_empty if count > 0]
    if len(non_empty) == 1:
        return non_empty[0][1]
    return None


def infer_doc_count(args: argparse.Namespace, data_prefix: Path) -> int:
    if args.doc_count is not None:
        if args.doc_count <= 0:
            raise ValueError("--doc-count must be positive")
        return args.doc_count

    for fn in (
        infer_doc_count_from_stats,
        infer_doc_count_from_indexed_dataset,
    ):
        count = fn(data_prefix)
        if count is not None and count > 0:
            return count

    count = infer_doc_count_from_jsonl(data_prefix, args.source_jsonl)
    if count is not None and count > 0:
        return count

    raise FileNotFoundError(
        "cannot infer document count; pass --doc-count or --source-jsonl explicitly"
    )


def index_path(data_prefix: Path, split_name: str, num_samples: int, seed: int, *, shuffle: bool, padded: bool) -> Path:
    filename = f"{data_prefix}_{split_name}_indexmap_{num_samples}ns_{seed}s"
    if padded:
        filename += "_padded_samples"
    if shuffle:
        filename += "_shuffle"
    filename += "_decoder_packed_idx.npy"
    return Path(filename)


def ready_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".ready")


def valid_index(path: Path, min_samples: int) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        arr = np.load(path, allow_pickle=True, mmap_mode="r")
        return arr.shape[0] >= min_samples
    except Exception:
        return False


def build_index(path: Path, docs: int, min_samples: int, seed: int, start_index: int, shuffle: bool) -> None:
    rng = np.random.RandomState(seed=seed)
    pieces = []
    total = 0
    while total < min_samples:
        ids = np.arange(start_index, start_index + docs, dtype=np.int64)
        if shuffle:
            rng.shuffle(ids)
        pieces.append(ids)
        total += ids.shape[0]

    arr = np.concatenate(pieces)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    np.save(tmp, arr, allow_pickle=True)
    os.replace(Path(str(tmp) + ".npy"), path)
    ready_path(path).write_text(str(time.time()), encoding="utf-8")


def wait_until_ready(path: Path, min_samples: int, timeout: int) -> None:
    deadline = time.time() + timeout
    marker = ready_path(path)
    while time.time() < deadline:
        if marker.exists() and valid_index(path, min_samples):
            return
        time.sleep(5)
    raise TimeoutError(f"timed out waiting for valid indexmap: {path}")


def main() -> int:
    args = parse_args()
    data_prefix = resolve_data_prefix(args)
    num_samples = resolve_num_samples(args)
    shuffle = not args.no_shuffle
    path = index_path(
        data_prefix,
        args.split_name,
        num_samples,
        args.seed,
        shuffle=shuffle,
        padded=args.padded_samples,
    )

    if args.wait:
        wait_until_ready(path, num_samples, args.timeout)
        print(f"[indexmap] ready: {path}")
        return 0

    docs = infer_doc_count(args, data_prefix)
    if valid_index(path, num_samples):
        ready_path(path).write_text(str(time.time()), encoding="utf-8")
        print(f"[indexmap] existing valid: {path}")
        return 0

    print(
        f"[indexmap] building {path} docs={docs} min_samples={num_samples} "
        f"seed={args.seed} shuffle={shuffle}"
    )
    build_index(path, docs, num_samples, args.seed, args.start_index, shuffle)
    print(f"[indexmap] built: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
