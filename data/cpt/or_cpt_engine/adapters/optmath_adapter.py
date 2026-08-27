from __future__ import annotations

import importlib.util
import inspect
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from or_cpt_engine.adapters.base import BaseORGeneratorAdapter
from or_cpt_engine.schemas.common import GeneratorRecord


class AdapterUnsupportedError(RuntimeError):
    pass


class OptMATHGeneratorAdapter(BaseORGeneratorAdapter):
    def __init__(self, registry_record: dict[str, Any] | GeneratorRecord):
        self.registry_record = (
            registry_record
            if isinstance(registry_record, GeneratorRecord)
            else GeneratorRecord.model_validate(registry_record)
        )
        self.module: Any | None = None
        self._generate_callable: Callable[..., Any] | None = None
        self._generator_class: type | None = None
        self._generator_method_name: str | None = None

    def load(self) -> None:
        if not self.registry_record.python_files:
            raise AdapterUnsupportedError("registry record has no python files")
        first_file = Path(self.registry_record.python_files[0])
        module_name = f"or_cpt_optmath_{self.registry_record.generator_id}_{int(time.time() * 1000)}"
        spec = importlib.util.spec_from_file_location(module_name, first_file)
        if spec is None or spec.loader is None:
            raise AdapterUnsupportedError(f"cannot import generator file: {first_file}")
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(first_file.parent))
        try:
            spec.loader.exec_module(module)
        finally:
            try:
                sys.path.remove(str(first_file.parent))
            except ValueError:
                pass
        self.module = module
        self._generate_callable = self._find_generate_callable(module)

    def generate(self, seed: int, difficulty_config: dict[str, Any]) -> dict[str, Any]:
        if self._generator_class is not None and self._generator_method_name is not None:
            generator = self._instantiate_generator(seed, difficulty_config)
            generated = getattr(generator, self._generator_method_name)()
            return self._wrap_generated(generated, generator)
        if self._generate_callable is None:
            raise AdapterUnsupportedError("no supported generate callable found")
        generated = self._call_generate(self._generate_callable, seed, difficulty_config)
        return self._wrap_generated(generated)

    def extract_model_info(self, raw_output: Any, *, solve_timeout_sec: int = 60) -> dict[str, Any]:
        payload = raw_output if isinstance(raw_output, dict) else {"raw_output_repr": repr(raw_output)}
        if payload.get("gurobi_model") is not None:
            return self._extract_gurobi_model_info(payload, solve_timeout_sec=solve_timeout_sec)
        code = payload.get("gurobi_code") or payload.get("code") or payload.get("python_code")
        return {
            "generation_params": payload.get("generation_params") or payload.get("params") or {},
            "model_type": payload.get("model_type"),
            "optimization_sense": payload.get("optimization_sense") or payload.get("sense"),
            "num_variables": payload.get("num_variables") or payload.get("n_variables"),
            "num_constraints": payload.get("num_constraints") or payload.get("n_constraints"),
            "lp_text": payload.get("lp_text") or payload.get("lp") or payload.get("lp_data"),
            "math_formula": payload.get("math_formula") or payload.get("formulation") or payload.get("mathematical_expression"),
            "gurobi_code": code,
            "initial_solver_status": payload.get("solver_status") or payload.get("status"),
            "initial_objective_value": payload.get("objective_value") or payload.get("obj_val") or payload.get("objective"),
        }

    def _find_generate_callable(self, module: Any) -> Callable[..., Any] | None:
        for name in ("generate_instance", "generate", "sample_instance", "main"):
            candidate = getattr(module, name, None)
            if callable(candidate):
                return candidate
        for _, candidate in inspect.getmembers(module, inspect.isclass):
            if candidate.__module__ != module.__name__:
                continue
            for method_name in ("generate_instance", "generate", "sample_instance"):
                if callable(getattr(candidate, method_name, None)):
                    self._generator_class = candidate
                    self._generator_method_name = method_name
                    return getattr(candidate, method_name)
        return None

    def _instantiate_generator(self, seed: int, difficulty_config: dict[str, Any]) -> Any:
        if self._generator_class is None:
            raise AdapterUnsupportedError("generator class is not loaded")
        init_signature = inspect.signature(self._generator_class)
        parameters = difficulty_config.get("parameters") if isinstance(difficulty_config, dict) else None
        attempts: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {}
        if "seed" in init_signature.parameters:
            kwargs["seed"] = seed
        if parameters is not None and "parameters" in init_signature.parameters:
            kwargs["parameters"] = parameters
        if kwargs:
            attempts.append(kwargs)
        if "seed" in init_signature.parameters:
            attempts.append({"seed": seed})
        attempts.append({})
        last_error: Exception | None = None
        for attempt in attempts:
            try:
                return self._generator_class(**attempt)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise AdapterUnsupportedError(f"cannot instantiate generator class: {last_error}")

    def _call_generate(self, func: Callable[..., Any], seed: int, difficulty_config: dict[str, Any]) -> Any:
        signature = inspect.signature(func)
        kwargs: dict[str, Any] = {}
        if "seed" in signature.parameters:
            kwargs["seed"] = seed
        if "difficulty_config" in signature.parameters:
            kwargs["difficulty_config"] = difficulty_config
        elif "config" in signature.parameters:
            kwargs["config"] = difficulty_config
        if kwargs:
            return func(**kwargs)
        if len(signature.parameters) >= 2:
            return func(seed, difficulty_config)
        if len(signature.parameters) == 1:
            return func(seed)
        return func()

    def _wrap_generated(self, generated: Any, generator_instance: Any | None = None) -> dict[str, Any]:
        if isinstance(generated, dict):
            payload = dict(generated)
        elif hasattr(generated, "to_dict") and callable(generated.to_dict):
            payload = dict(generated.to_dict())
        elif _is_gurobi_model(generated):
            payload = {"gurobi_model": generated}
        else:
            payload = {"raw_output_repr": repr(generated)}
        if generator_instance is not None:
            payload["generator_attrs"] = _extract_generator_attrs(generator_instance)
        return payload

    def _extract_gurobi_model_info(self, payload: dict[str, Any], *, solve_timeout_sec: int = 60) -> dict[str, Any]:
        import logging

        model = payload["gurobi_model"]
        generator_attrs = payload.get("generator_attrs") or {}
        generator_id = generator_attrs.get("generator_id") or self.registry_record.generator_id or "unknown"
        try:
            model.Params.OutputFlag = 0
        except Exception:
            pass
        try:
            model.update()
        except Exception:
            pass
        try:
            if getattr(model.Params, "TimeLimit", 0) >= 1e9 or getattr(model.Params, "TimeLimit", 0) <= 0:
                model.Params.TimeLimit = float(solve_timeout_sec)
            model.optimize()
            status = _gurobi_status_name(model)
            if status == "TIME_LIMIT":
                logging.getLogger("or_cpt_engine.solver").warning(
                    "solver_exec status=TIME_LIMIT generator_id=%s objective_value=%s",
                    generator_id,
                    float(model.ObjVal) if getattr(model, "SolCount", 0) > 0 else "none",
                )
        except Exception:
            pass
        lp_text = _write_model_to_lp(model)
        status = _gurobi_status_name(model)
        objective_value = None
        try:
            if getattr(model, "SolCount", 0) > 0:
                objective_value = float(model.ObjVal)
        except Exception:
            objective_value = None
        return {
            "generation_params": generator_attrs.get("parameters") or {},
            "model_type": self.registry_record.model_type,
            "optimization_sense": _objective_sense(model) or self.registry_record.optimization_sense,
            "num_variables": _safe_int_attr(model, "NumVars"),
            "num_constraints": _safe_int_attr(model, "NumConstrs"),
            "lp_text": lp_text,
            "math_formula": (
                generator_attrs.get("mathematical_formulation")
                or self.registry_record.metadata.get("math_formula")
                or self.registry_record.metadata.get("mathematical_expression")
            ),
            "gurobi_code": None,
            "initial_solver_status": status,
            "initial_objective_value": objective_value,
        }


def _is_gurobi_model(value: Any) -> bool:
    try:
        import gurobipy as gp  # type: ignore

        return isinstance(value, gp.Model)
    except Exception:
        return False


def _extract_generator_attrs(generator: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("problem_type", "mathematical_formulation", "parameters"):
        value = getattr(generator, name, None)
        if isinstance(value, (str, int, float, bool, list, tuple, dict)) or value is None:
            result[name] = value
    return result


def _write_model_to_lp(model: Any) -> str | None:
    with tempfile.TemporaryDirectory(prefix="or_cpt_optmath_model_") as temp_dir:
        lp_path = Path(temp_dir) / "instance.lp"
        try:
            model.write(str(lp_path))
            return lp_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None


def _gurobi_status_name(model: Any) -> str | None:
    try:
        import gurobipy as gp  # type: ignore

        status_map = {
            gp.GRB.OPTIMAL: "OPTIMAL",
            gp.GRB.INFEASIBLE: "INFEASIBLE",
            gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
            gp.GRB.UNBOUNDED: "UNBOUNDED",
            gp.GRB.TIME_LIMIT: "TIME_LIMIT",
            gp.GRB.LOADED: "LOADED",
        }
        return status_map.get(model.Status, str(model.Status))
    except Exception:
        return None


def _objective_sense(model: Any) -> str | None:
    try:
        return "minimize" if int(model.ModelSense) == 1 else "maximize"
    except Exception:
        return None


def _safe_int_attr(model: Any, attr_name: str) -> int | None:
    try:
        return int(getattr(model, attr_name))
    except Exception:
        return None
