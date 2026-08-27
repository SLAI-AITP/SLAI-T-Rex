from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel

_VERSION_RE = re.compile(r"^v(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")
DEFAULT_PROMPT_VERSIONS_PATH = Path(__file__).resolve().parents[1] / "prompts" / "prompt_versions.yaml"


class PromptSpec(BaseModel):
    version: str
    path: str
    updated_at: str | None = None
    notes: str | None = None
    hash: str
    text: str


class PromptRegistry:
    def __init__(self, prompt_versions_path: str | Path = DEFAULT_PROMPT_VERSIONS_PATH):
        self._versions_path = Path(prompt_versions_path).resolve()
        self._raw_configs: dict[str, dict] = {}
        self._prompts: dict[str, PromptSpec] = {}
        self.reload()

    def reload(self) -> None:
        with self._versions_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        prompts: dict[str, PromptSpec] = {}
        raw_configs: dict[str, dict] = {}
        for prompt_name, config in raw.items():
            prompt_path = (self._versions_path.parent / str(config["path"])).resolve()
            text = prompt_path.read_text(encoding="utf-8")
            prompts[prompt_name] = PromptSpec(
                version=str(config["version"]),
                path=str(prompt_path),
                updated_at=config.get("updated_at"),
                notes=config.get("notes"),
                hash=sha256_text(text),
                text=text,
            )
            raw_configs[prompt_name] = dict(config)

        self._prompts = prompts
        self._raw_configs = raw_configs

    def get(self, prompt_name: str) -> PromptSpec:
        return self._prompts[prompt_name]

    def promote(
        self,
        prompt_name: str,
        *,
        text: str,
        notes: str,
        version: str | None = None,
    ) -> tuple[PromptSpec, PromptSpec]:
        previous_spec = self.get(prompt_name).model_copy(deep=True)
        next_version = version or bump_patch_version(previous_spec.version)
        prompt_path = Path(previous_spec.path)
        prompt_path.write_text(text, encoding="utf-8")

        raw_config = dict(self._raw_configs[prompt_name])
        raw_config["version"] = next_version
        raw_config["updated_at"] = timestamp_now()
        raw_config["notes"] = notes
        self._raw_configs[prompt_name] = raw_config
        self._write_versions_file()
        self.reload()
        return previous_spec, self.get(prompt_name).model_copy(deep=True)

    def _write_versions_file(self) -> None:
        serializable = {
            prompt_name: {
                "version": config["version"],
                "path": config["path"],
                "updated_at": config.get("updated_at"),
                "notes": config.get("notes"),
            }
            for prompt_name, config in self._raw_configs.items()
        }
        with self._versions_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(serializable, handle, sort_keys=False, allow_unicode=True)


def load_default_prompt_registry() -> PromptRegistry:
    return PromptRegistry(DEFAULT_PROMPT_VERSIONS_PATH)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bump_patch_version(version: str) -> str:
    match = _VERSION_RE.match(version)
    if match is None:
        return f"{version}.1"
    major = int(match.group("major"))
    minor = int(match.group("minor"))
    patch = int(match.group("patch")) + 1
    return f"v{major}.{minor}.{patch}"


def timestamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
