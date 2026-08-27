from __future__ import annotations

import importlib.util
import os
import platform
import sys
from pathlib import Path
from typing import Any

import yaml

from cpt_cleaner.utils import json_compat

from or_cpt_engine.schemas.common import EngineConfig
from or_cpt_engine.utils.io import write_text


def run_bootstrap_checks(config: EngineConfig, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checks = [
        _check_python_version(),
        _check_optmath_paths(config),
        _check_gurobi(),
        _check_llm_env(config),
        _check_production_configs(config),
        _check_output_dirs(config),
    ]
    errors = [error for check in checks for error in check.get("errors", [])]
    warnings = [warning for check in checks for warning in check.get("warnings", [])]
    report = {
        "status": "PASS" if not errors else "FAIL",
        "python": sys.version,
        "platform": platform.platform(),
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }
    (output / "environment_report.json").write_bytes(json_compat.dumps(report))
    write_text(output / "environment_report.md", render_environment_report(report))
    return report


def render_environment_report(report: dict[str, Any]) -> str:
    lines = [
        "# OR-CPT Environment Report",
        "",
        f"- Status: {report['status']}",
        f"- Python: {report['python'].split()[0]}",
        f"- Platform: {report['platform']}",
        "",
        "## Checks",
        "",
        "| Check | Status | Details |",
        "|---|---|---|",
    ]
    for check in report["checks"]:
        detail_parts = []
        if check.get("details"):
            detail_parts.append(str(check["details"]).replace("|", "\\|"))
        if check.get("warnings"):
            detail_parts.append("warnings=" + "; ".join(check["warnings"]).replace("|", "\\|"))
        if check.get("errors"):
            detail_parts.append("errors=" + "; ".join(check["errors"]).replace("|", "\\|"))
        lines.append(f"| {check['name']} | {check['status']} | {'<br>'.join(detail_parts)} |")
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def _check_python_version() -> dict[str, Any]:
    errors: list[str] = []
    if sys.version_info < (3, 10):
        errors.append("Python >= 3.10 is required for the current package metadata.")
    warnings: list[str] = []
    if sys.version_info[:2] != (3, 11):
        warnings.append("Python 3.11 is recommended for production OR-CPT runs.")
    return {
        "name": "python",
        "status": "PASS" if not errors else "FAIL",
        "details": {"version": sys.version.split()[0]},
        "warnings": warnings,
        "errors": errors,
    }


def _check_optmath_paths(config: EngineConfig) -> dict[str, Any]:
    root = Path(config.paths.external_optmath_root)
    generators = Path(config.paths.optmath_generators_dir)
    errors: list[str] = []
    warnings: list[str] = []
    if not root.exists():
        errors.append(f"OptMATH root does not exist: {root}")
    if not generators.exists():
        errors.append(f"OptMATH generators dir does not exist: {generators}")
    elif not any(generators.iterdir()):
        warnings.append(f"OptMATH generators dir is empty: {generators}")
    return {
        "name": "optmath_paths",
        "status": "PASS" if not errors else "FAIL",
        "details": {"root": str(root), "generators": str(generators)},
        "warnings": warnings,
        "errors": errors,
    }


def _check_gurobi() -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {"importable": False}
    if importlib.util.find_spec("gurobipy") is None:
        errors.append("gurobipy is not importable.")
    else:
        details["importable"] = True
        try:
            import gurobipy as gp  # type: ignore

            model = gp.Model()
            model.Params.OutputFlag = 0
            details["license_check"] = "model_created"
        except Exception as exc:  # noqa: BLE001
            errors.append(f"gurobipy import succeeded but model creation failed: {exc}")
    return {
        "name": "gurobi",
        "status": "PASS" if not errors else "FAIL",
        "details": details,
        "warnings": warnings,
        "errors": errors,
    }


def _check_llm_env(config: EngineConfig) -> dict[str, Any]:
    errors: list[str] = []
    details: dict[str, Any] = {}
    for name, provider in config.llm.providers.items():
        if provider.endpoint_pool_path:
            pool_path = Path(provider.endpoint_pool_path)
            endpoint_count = 0
            if not pool_path.exists():
                errors.append(f"LLM provider {name} endpoint pool not found: {pool_path}")
            else:
                try:
                    with pool_path.open("r", encoding="utf-8") as handle:
                        payload = yaml.safe_load(handle) or {}
                    endpoint_count = len(payload.get("endpoints") or [])
                    if endpoint_count == 0:
                        errors.append(f"LLM provider {name} endpoint pool is empty: {pool_path}")
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"LLM provider {name} endpoint pool cannot be read: {exc}")
            details[name] = {
                "mode": "endpoint_pool",
                "endpoint_pool_path": str(pool_path),
                "endpoint_count": endpoint_count,
                "model_default": provider.model,
                "concurrency_per_endpoint": provider.concurrency_per_endpoint,
                "effective_total_llm_concurrency": endpoint_count * provider.concurrency_per_endpoint,
            }
            continue
        base_url = os.getenv(provider.base_url_env)
        api_key = os.getenv(provider.api_key_env)
        details[name] = {
            "mode": "env_single_endpoint",
            "base_url_env": provider.base_url_env,
            "base_url_set": bool(base_url),
            "api_key_env": provider.api_key_env,
            "api_key_set": bool(api_key),
            "model": provider.model,
        }
        if not base_url:
            errors.append(f"LLM provider {name} missing env var {provider.base_url_env}")
    return {
        "name": "llm_env",
        "status": "PASS" if not errors else "FAIL",
        "details": details,
        "warnings": [],
        "errors": errors,
    }


def _check_output_dirs(config: EngineConfig) -> dict[str, Any]:
    paths = [config.paths.runs_root, config.paths.registry_root, config.paths.final_root]
    errors: list[str] = []
    created: list[str] = []
    for path in paths:
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            created.append(str(path))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"cannot create output directory {path}: {exc}")
    return {
        "name": "output_dirs",
        "status": "PASS" if not errors else "FAIL",
        "details": {"created_or_existing": created},
        "warnings": [],
        "errors": errors,
    }


def _check_production_configs(config: EngineConfig) -> dict[str, Any]:
    required_paths = {
        "scenario_catalog": config.paths.scenario_catalog_path,
        "generator_profiles": config.paths.generator_profiles_path,
        "distribution_policy": config.paths.distribution_policy_path,
        "quality_gate": config.paths.quality_gate_path,
    }
    errors: list[str] = []
    details: dict[str, str] = {}
    for name, path_value in required_paths.items():
        path = Path(path_value)
        details[name] = str(path)
        if not path.exists():
            errors.append(f"{name} config does not exist: {path}")
    return {
        "name": "production_configs",
        "status": "PASS" if not errors else "FAIL",
        "details": details,
        "warnings": [],
        "errors": errors,
    }
