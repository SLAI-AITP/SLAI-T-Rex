from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from or_cpt_engine.utils.io import write_jsonl, write_text


EXCLUDED_DIR_NAMES = {"__pycache__", ".git", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
PROVENANCE_FILENAME = "provenance_manifest.jsonl"
INTERNAL_README_FILENAME = "README.md"


def import_optmath_generators(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(source_dir)
    output = Path(output_dir)
    if not source.exists():
        raise FileNotFoundError(f"OptMATH generator source directory not found: {source}")
    generator_dirs = sorted(path for path in source.iterdir() if path.is_dir())
    imported_at = datetime.now(timezone.utc).isoformat()

    manifest_rows: list[dict[str, Any]] = []
    skipped_existing = 0
    copied_files = 0
    planned_files = 0

    for generator_dir in generator_dirs:
        target_dir = output / generator_dir.name
        files = [path for path in sorted(generator_dir.rglob("*")) if _should_copy_file(path)]
        file_rows: list[dict[str, Any]] = []
        for source_file in files:
            relative_path = source_file.relative_to(generator_dir).as_posix()
            target_file = target_dir / relative_path
            file_rows.append(
                {
                    "relative_path": relative_path,
                    "source_sha256": _sha256_file(source_file),
                    "copied_sha256": _sha256_file(source_file),
                    "size_bytes": source_file.stat().st_size,
                }
            )
            planned_files += 1
            if dry_run:
                continue
            if target_file.exists() and not overwrite:
                skipped_existing += 1
                continue
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)
            copied_files += 1

        manifest_rows.append(
            {
                "generator_name": generator_dir.name,
                "generator_id": _safe_generator_id(generator_dir.name),
                "source": "optmath_internal",
                "upstream_path": str(generator_dir),
                "internal_path": str(target_dir),
                "imported_at": imported_at,
                "overwrite": overwrite,
                "dry_run": dry_run,
                "local_modified": False,
                "file_count": len(file_rows),
                "files": file_rows,
            }
        )

    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)
        write_jsonl(output / PROVENANCE_FILENAME, manifest_rows)
        _copy_license_and_notice(source, output)
        write_text(output / INTERNAL_README_FILENAME, _render_internal_readme(source, imported_at, len(generator_dirs)))

    return {
        "source": str(source),
        "output": str(output),
        "dry_run": dry_run,
        "overwrite": overwrite,
        "generators": len(generator_dirs),
        "planned_files": planned_files,
        "copied_files": copied_files,
        "skipped_existing_files": skipped_existing,
        "provenance_path": str(output / PROVENANCE_FILENAME),
    }


def _copy_license_and_notice(source: Path, output: Path) -> None:
    optmath_root = source.parent
    for filename in ("LICENSE", "NOTICE"):
        source_file = optmath_root / filename
        if source_file.exists():
            shutil.copy2(source_file, output / f"UPSTREAM_{filename}")
    pyproject = optmath_root / "pyproject.toml"
    if pyproject.exists():
        shutil.copy2(pyproject, output / "UPSTREAM_pyproject.toml")


def _render_internal_readme(source: Path, imported_at: str, generator_count: int) -> str:
    return "\n".join(
        [
            "# Internal OptMATH Seed Generators",
            "",
            "This directory contains a project-local copy of the OptMATH seed generators used by `or_cpt_engine`.",
            "",
            f"- Upstream source directory: `{source}`",
            f"- Imported at: `{imported_at}`",
            f"- Generator folders imported: `{generator_count}`",
            "- Upstream license files are preserved as `UPSTREAM_LICENSE` and, when available, `UPSTREAM_NOTICE`.",
            "- `provenance_manifest.jsonl` records file checksums captured at import time.",
            "- Future local generator improvements should be made here, not under `external/OptMATH`.",
            "- When modifying an imported generator, also update the corresponding generator profile and quality notes.",
            "",
        ]
    )


def _should_copy_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_generator_id(name: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return f"optmath_{normalized or 'generator'}"

