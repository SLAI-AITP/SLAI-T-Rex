from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from or_cpt_engine.schemas.common import GeneratorRecord
from or_cpt_engine.utils.io import write_jsonl, write_text


def _safe_generator_id(name: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return f"optmath_{normalized or 'generator'}"


def _read_json(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.exists():
        return {}, ["metadata.json not found"]
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except json.JSONDecodeError as exc:
        return {}, [f"metadata.json invalid: {exc.msg}"]


def _read_readme_excerpt(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return text[:1200] if text else None


def _infer_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def scan_optmath_generators(
    generators_dir: str | Path,
    output_dir: str | Path,
    *,
    source_label: str | None = None,
    profiles_path: str | Path | None = None,
) -> dict[str, int]:
    source_dir = Path(generators_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"OptMATH generators directory not found: {source_dir}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    registered: list[GeneratorRecord] = []
    invalid: list[dict[str, Any]] = []
    provenance_by_name = _load_provenance(source_dir)
    profile_ids = _load_profile_ids(profiles_path)
    resolved_source_label = source_label or _infer_source_label(source_dir)

    for child in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        metadata_path = child / "metadata.json"
        readme_path = child / "readme.md"
        if not readme_path.exists():
            readme_candidates = list(child.glob("README*")) + list(child.glob("Readme*"))
            if readme_candidates:
                readme_path = readme_candidates[0]

        metadata, errors = _read_json(metadata_path)
        python_files = sorted(str(path) for path in child.glob("*.py") if not path.name.startswith("__"))
        if not python_files:
            errors.append("no python file found")

        generator_id = _safe_generator_id(child.name)
        provenance = provenance_by_name.get(child.name) or provenance_by_name.get(generator_id) or {}
        record = GeneratorRecord(
            generator_id=generator_id,
            source=resolved_source_label,
            name=child.name,
            path=str(child),
            python_files=python_files,
            metadata_path=str(metadata_path) if metadata_path.exists() else None,
            readme_path=str(readme_path) if readme_path.exists() else None,
            metadata=metadata,
            readme_excerpt=_read_readme_excerpt(readme_path),
            domain=_infer_list(metadata.get("domain") or metadata.get("domains")),
            model_type=metadata.get("model_type") or metadata.get("problem_type"),
            optimization_sense=metadata.get("optimization_sense") or metadata.get("sense"),
            has_metadata=metadata_path.exists() and not any("metadata.json invalid" in error for error in errors),
            has_readme=readme_path.exists(),
            has_python_generator=bool(python_files),
            status="INVALID" if errors else "REGISTERED",
            errors=errors,
            upstream_generator_name=provenance.get("generator_name") or child.name,
            upstream_path=provenance.get("upstream_path"),
            internal_path=str(child) if resolved_source_label == "optmath_internal" else None,
            profile_status="PROFILED" if generator_id in profile_ids else "MISSING",
            local_modified=_is_locally_modified(child, provenance),
        )
        if errors:
            invalid.append(record.model_dump(mode="json"))
        else:
            registered.append(record)

    write_jsonl(output / "generator_registry.jsonl", registered)
    write_jsonl(output / "invalid_generators.jsonl", invalid)
    write_text(output / "generator_inventory.md", render_generator_inventory(registered, invalid))
    return {
        "scanned": len(registered) + len(invalid),
        "registered": len(registered),
        "invalid": len(invalid),
    }


def render_generator_inventory(registered: list[GeneratorRecord], invalid: list[dict[str, Any]]) -> str:
    lines = [
        "# Generator Inventory",
        "",
        "## Summary",
        "",
        f"- Total generator folders scanned: {len(registered) + len(invalid)}",
        f"- Registered generators: {len(registered)}",
        f"- Invalid generators: {len(invalid)}",
        "",
        "## Registered Generators",
        "",
        "| generator_id | name | source | model_type | profile | local_modified | status |",
        "|---|---|---|---|---|---:|---|",
    ]
    for record in registered:
        lines.append(
            f"| {record.generator_id} | {record.name} | {record.source} | {record.model_type or ''} | "
            f"{record.profile_status} | {str(record.local_modified).lower()} | {record.status} |"
        )
    if invalid:
        lines.extend(["", "## Invalid Generators", "", "| name | errors |", "|---|---|"])
        for record in invalid:
            lines.append(f"| {record.get('name')} | {'; '.join(record.get('errors', []))} |")
    return "\n".join(lines) + "\n"


def _infer_source_label(source_dir: Path) -> str:
    normalized = source_dir.as_posix()
    return "optmath_internal" if "or_cpt_engine/generators/optmath_seed" in normalized else "optmath"


def _load_profile_ids(profiles_path: str | Path | None) -> set[str]:
    if profiles_path is None:
        return set()
    path = Path(profiles_path)
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    profiles = payload.get("profiles") if isinstance(payload, dict) else None
    return set(profiles) if isinstance(profiles, dict) else set()


def _load_provenance(source_dir: Path) -> dict[str, dict[str, Any]]:
    path = source_dir / "provenance_manifest.jsonl"
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open("rb") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            result[str(row.get("generator_name"))] = row
            result[str(row.get("generator_id"))] = row
    return result


def _is_locally_modified(generator_dir: Path, provenance: dict[str, Any]) -> bool:
    # The provenance manifest can explicitly mark a generator as locally tuned
    # relative to upstream OptMATH. Keep that semantic flag even when the current
    # files still match the locally captured checksum.
    if bool(provenance.get("local_modified")):
        return True

    files = provenance.get("files")
    if not isinstance(files, list) or not files:
        return False
    for file_row in files:
        if not isinstance(file_row, dict):
            continue
        relative_path = file_row.get("relative_path")
        expected_hash = file_row.get("copied_sha256")
        if not relative_path or not expected_hash:
            continue
        current_path = generator_dir / str(relative_path)
        if not current_path.exists():
            return True
        if _sha256_file(current_path) != expected_hash:
            return True
    return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
