import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set

from tqdm import tqdm

from evaluation.common.fewshot_acc import (
    attempt_seed,
    augment_prompt_text,
    fewshot_metadata,
    load_result_fewshot_pool,
    load_sft_fewshot_pool,
    select_fewshot_examples,
    select_similar_fewshot_examples,
)


PROMPT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "prompt_template_lp_structure.txt"
)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
LP_UTILS_PATH = os.path.join(REPO_ROOT, "evaluate", "_lp_utils.py")
EQUIV_CHECK_PATH = os.path.join(REPO_ROOT, "evaluate", "equivalence_check.py")

_thread_local = threading.local()
_eval_semaphore = None


def load_module_from_path(module_name: str, module_path: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_lp_utils_mod = load_module_from_path("bench4opt_lp_utils", LP_UTILS_PATH)
extract_python_code = _lp_utils_mod.extract_python_code
ensure_imports = _lp_utils_mod.ensure_imports
process_code_for_lp = _lp_utils_mod.process_code_for_lp
_equiv_mod = load_module_from_path("bench4opt_equivcheck", EQUIV_CHECK_PATH)
check_lp_equivalence_fast = _equiv_mod.check_lp_equivalence_fast


def load_prompt_template() -> str:
    with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as handle:
        return handle.read()


def get_gurobi_version() -> str:
    try:
        import gurobipy as gp

        return ".".join(str(part) for part in gp.gurobi.version())
    except Exception:
        return "unknown"


def render_prompt(template: str, sample_data: dict) -> str:
    prompt = template.replace("[Problem Description]", sample_data["problem"])
    prompt = prompt.replace("[Gurobi Version]", get_gurobi_version())
    return prompt


def _evaluate_completion_worker(data_dir: str, sample_data: dict, model_name: str, queue: mp.Queue):
    try:
        result = evaluate_completion(data_dir, sample_data, model_name)
        queue.put((True, result))
    except Exception as exc:
        queue.put((False, str(exc)))


def evaluate_completion_with_timeout(data_dir: str, sample_data: dict, model_name: str, timeout_seconds: int):
    if timeout_seconds <= 0:
        return evaluate_completion(data_dir, sample_data, model_name)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_evaluate_completion_worker,
        args=(data_dir, sample_data, model_name, queue),
    )
    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise TimeoutError(f"evaluation timeout after {timeout_seconds}s")

    if queue.empty():
        raise RuntimeError("evaluation process finished without returning result")

    success, payload = queue.get()
    if not success:
        raise RuntimeError(payload)
    return payload


def load_samples(data_dir: str) -> List[dict]:
    test_path = os.path.join(data_dir, "test.jsonl")
    with open(test_path, "r", encoding="utf-8") as handle:
        raw_samples = [json.loads(line) for line in handle]

    samples = []
    for sample in raw_samples:
        samples.append(
            {
                "id": sample["id"],
                "data_path": sample["data_path"],
                "problem": sample["problem"],
                "reference_lp_path": sample["reference_lp_path"],
                "reference_code": sample.get("reference_code"),
                "problem_type": sample.get("problem_type"),
                "problem_class": sample.get("problem_class"),
            }
        )
    return samples


def get_openai_client(api_key: str, base_url: str):
    client = getattr(_thread_local, "openai_client", None)
    client_key = getattr(_thread_local, "openai_client_key", None)
    client_base_url = getattr(_thread_local, "openai_client_base_url", None)

    if client is None or client_key != api_key or client_base_url != base_url:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        _thread_local.openai_client = client
        _thread_local.openai_client_key = api_key
        _thread_local.openai_client_base_url = base_url
    return client


def generate_completion(
    prompt: str,
    model_name: str,
    openai_api_key: str,
    openai_base_url: str,
    **generation_kwargs,
) -> str:
    client = get_openai_client(openai_api_key, openai_base_url)

    request_kwargs = {
        "model": model_name,
        "seed": 42,
    }
    request_kwargs.update({k: v for k, v in generation_kwargs.items() if v is not None})

    from openai import APIError, InternalServerError

    for attempt in range(3):
        try:
            request_output = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                **request_kwargs,
            )
            return request_output.choices[0].message.content or ""
        except (InternalServerError, APIError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def convert_to_lp(
    model_name: str,
    data_dir: str,
    code: str,
    data_path: str,
    problem_id: str,
    verbose: bool = False,
):
    data_dir_name = data_dir.split("/")[-1]
    temp_dir = f"temp_lp/{model_name}_{data_dir_name}_lp/"
    os.makedirs(temp_dir, exist_ok=True)

    lp_file_name = f"{problem_id}_model_{data_path.split('/')[-1].split('.')[0]}.lp"
    lp_file_path = os.path.join(temp_dir, lp_file_name)
    if os.path.exists(lp_file_path):
        os.remove(lp_file_path)

    if verbose:
        print(f"Processing code for to write LP file at: {lp_file_path}")

    extracted_code = extract_python_code(code)
    code_for_lp = process_code_for_lp(extracted_code, data_path, lp_file_path)
    code_for_lp = ensure_imports(code_for_lp)

    os.makedirs(f"temp_code/{model_name}/", exist_ok=True)
    temp_exec_path = f"temp_code/{model_name}/temp_exec_{problem_id}.py"
    with open(temp_exec_path, "w", encoding="utf-8") as handle:
        handle.write(code_for_lp)

    result = subprocess.run(
        [sys.executable, temp_exec_path],
        capture_output=True,
        text=True,
    )

    if os.path.exists(lp_file_path):
        with open(lp_file_path, "r", encoding="utf-8") as handle:
            lp_content = handle.read()
        return True, lp_content, lp_file_path

    error_msg = result.stderr.strip() if result.stderr.strip() else "Unknown error"
    return False, "LP file was not created due to error: " + error_msg, None


def evaluate_generated_code(
    model_name: str,
    data_dir: str,
    code: str,
    data_path: str,
    reference_lp_path: str,
    problem_id: str,
    verbose: bool = False,
):
    if not reference_lp_path or not os.path.exists(reference_lp_path):
        code_eval_result = {
            "success": False,
            "message": "no reference lp",
            "lp_file_path": "",
        }
        equivalence_eval_result = {"success": False, "message": ""}
        return code_eval_result, equivalence_eval_result, ""

    code_success, code_result, lp_file_path = convert_to_lp(
        model_name,
        data_dir,
        code,
        data_path,
        verbose=verbose,
        problem_id=problem_id,
    )

    if not code_success:
        code_eval_result = {
            "success": False,
            "message": code_result,
            "lp_file_path": lp_file_path,
        }
        equivalence_eval_result = {"success": False, "message": ""}
        return code_eval_result, equivalence_eval_result, ""

    start_time = time.time()
    is_equivalent, error_msg, time_info = check_lp_equivalence_fast(reference_lp_path, lp_file_path)
    end_time = time.time()
    equivalence_check_time = {
        "total_time": end_time - start_time,
        "step_time": time_info,
    }

    code_eval_result = {
        "success": True,
        "message": code_result,
        "lp_file_path": lp_file_path,
    }
    equivalence_eval_result = {
        "success": is_equivalent,
        "message": error_msg,
    }
    return code_eval_result, equivalence_eval_result, equivalence_check_time


def evaluate_completion(data_dir: str, sample_data: dict, model_name: str):
    code = extract_python_code(sample_data["completion"])
    data_path = os.path.join(data_dir, sample_data["data_path"])
    reference_lp_path = os.path.join(data_dir, sample_data["reference_lp_path"])

    code_eval_result, wl_eval_result, equivalence_check_time = evaluate_generated_code(
        model_name=model_name,
        data_dir=data_dir,
        code=code,
        data_path=data_path,
        reference_lp_path=reference_lp_path,
        verbose=False,
        problem_id=sample_data["id"].split("_")[1],
    )

    if code_eval_result["success"]:
        code_success = True
        code_error_msg = ""
        code = code_eval_result["lp_file_path"]
        wl_success = wl_eval_result["success"]
        wl_error_msg = wl_eval_result["message"]
    else:
        code_success = False
        code_error_msg = code_eval_result["message"]
        code = code_eval_result["lp_file_path"]
        wl_success = False
        wl_error_msg = ""

    reward = {
        "code_reward": 1.0 if code_success else 0.0,
        "wl_reward": 1.0 if wl_success else 0.0,
    }
    verification = {
        "code_verification": code_error_msg,
        "wl_verification": wl_error_msg,
    }
    return code, reward, verification, equivalence_check_time


def run_sample(
    data_dir: str,
    sample_data: dict,
    model_name: str,
    prompt_template: str,
    generation_kwargs: dict,
    eval_timeout_seconds: int,
    openai_api_key: str,
    openai_base_url: str,
    few_shot_pool: List[dict] | None = None,
    few_shot: int = 0,
    few_shot_seed: int = 42,
    few_shot_strategy: str = "random",
    acc_samples: int = 1,
) -> dict:
    result = dict(sample_data)
    if few_shot_strategy == "similar":
        selected_examples = select_similar_fewshot_examples(
            few_shot_pool or [], result, few_shot, few_shot_seed
        )
    else:
        selected_examples = select_fewshot_examples(
            few_shot_pool or [], result["id"], few_shot, few_shot_seed
        )
    result["prompt"] = augment_prompt_text(render_prompt(prompt_template, result), selected_examples)

    attempts = []
    requested_attempts = max(1, int(acc_samples))
    base_seed = int(generation_kwargs.get("seed", 42))
    selected_attempt = None

    for attempt_index in range(requested_attempts):
        attempt_kwargs = dict(generation_kwargs)
        attempt_kwargs["seed"] = attempt_seed(base_seed, attempt_index)
        completion = generate_completion(
            result["prompt"],
            model_name,
            openai_api_key,
            openai_base_url,
            **attempt_kwargs,
        )

        attempt_data = dict(result)
        attempt_data["completion"] = completion
        try:
            if _eval_semaphore is not None:
                with _eval_semaphore:
                    code, reward, verification, equivalence_check_time = evaluate_completion_with_timeout(
                        data_dir,
                        attempt_data,
                        model_name,
                        eval_timeout_seconds,
                    )
            else:
                code, reward, verification, equivalence_check_time = evaluate_completion_with_timeout(
                    data_dir,
                    attempt_data,
                    model_name,
                    eval_timeout_seconds,
                )
        except Exception as exc:
            code = None
            reward = {"code_reward": 0.0, "wl_reward": 0.0}
            verification = {
                "code_verification": f"exception_or_timeout: {str(exc)}",
                "wl_verification": "",
            }
            equivalence_check_time = ""

        attempt_result = {
            "attempt_index": attempt_index,
            "seed": attempt_kwargs["seed"],
            "completion": completion,
            "code": code,
            "reward": reward,
            "verification": verification,
            "equivalence_check_time": equivalence_check_time,
        }
        attempts.append(attempt_result)
        if float(reward.get("wl_reward", 0.0)) >= 1.0:
            selected_attempt = attempt_result
            break

    if selected_attempt is None:
        selected_attempt = attempts[-1]

    result["completion"] = selected_attempt["completion"]
    result["code"] = selected_attempt["code"]
    result["reward"] = selected_attempt["reward"]
    result["verification"] = selected_attempt["verification"]
    result["equivalence_check_time"] = selected_attempt["equivalence_check_time"]
    result["acc_samples_requested"] = requested_attempts
    result["acc_attempts_run"] = len(attempts)
    result["acc_success_attempt"] = selected_attempt["attempt_index"] if float(selected_attempt["reward"].get("wl_reward", 0.0)) >= 1.0 else None
    result["attempts"] = attempts
    result.update(fewshot_metadata(selected_examples))
    result["few_shot_strategy"] = few_shot_strategy
    return result


def load_existing_results(save_path: str) -> Dict[str, dict]:
    if not os.path.exists(save_path):
        print(f"结果文件 {save_path} 不存在，将从头开始")
        return {}

    try:
        with open(save_path, "r", encoding="utf-8") as handle:
            results = json.load(handle)
    except Exception as exc:
        print(f"加载结果文件时出错: {exc}")
        return {}

    completed_samples = {}
    for result in results:
        sample_id = result.get("id")
        if sample_id:
            completed_samples[sample_id] = result

    print(f"成功加载 {len(completed_samples)} 个已完成的结果")
    return completed_samples


def save_results_incremental(all_results: List[dict], save_path: str, lock: threading.Lock):
    with lock:
        backup_path = save_path + ".backup"
        try:
            if os.path.exists(save_path):
                os.replace(save_path, backup_path)

            with open(save_path, "w", encoding="utf-8") as handle:
                json.dump(all_results, handle, ensure_ascii=False, indent=4)

            if os.path.exists(backup_path):
                os.remove(backup_path)
            print(f"结果已保存到 {save_path}")
        except Exception as exc:
            print(f"保存结果时出错: {exc}")
            if os.path.exists(backup_path):
                os.replace(backup_path, save_path)


def sort_results_by_numeric_id(results_by_id: Dict[str, dict]) -> List[dict]:
    def sample_sort_key(sample_id: str):
        try:
            return int(sample_id.split("_")[1])
        except (IndexError, ValueError):
            return sample_id

    return [results_by_id[sample_id] for sample_id in sorted(results_by_id.keys(), key=sample_sort_key)]


def filter_pending_samples(full_data: List[dict], completed_sample_ids: Set[str]) -> List[dict]:
    pending_samples = [sample for sample in full_data if sample["id"] not in completed_sample_ids]
    print(f"总样本数: {len(full_data)}, 已完成: {len(completed_sample_ids)}, 待处理: {len(pending_samples)}")
    return pending_samples


def main():
    global _eval_semaphore

    parser = argparse.ArgumentParser(description="Fast resume-capable Bench4Opt evaluation")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="Name of the tested model")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save results")
    parser.add_argument("--data_dir", type=str, default="data/bench4opt_mix", help="Path to data directory")
    parser.add_argument("--openai_api_key", type=str, default=None, help="API key for the model service")
    parser.add_argument("--openai_base_url", type=str, default=None, help="Base URL for the model service")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=None, help="Sampling top_p")
    parser.add_argument("--max_tokens", type=int, default=None, help="Maximum completion tokens")
    parser.add_argument("--max_workers", type=int, default=min(8, os.cpu_count() or 4), help="Maximum worker threads")
    parser.add_argument("--eval_workers", type=int, default=1, help="Maximum concurrent evaluation workers (recommended: 1)")
    parser.add_argument("--eval_timeout", type=int, default=180, help="Timeout in seconds for each sample evaluation")
    parser.add_argument("--save_every", type=int, default=16, help="Save after every N completed samples")
    parser.add_argument("--start", type=int, default=0, help="Start index in the loaded dataset")
    parser.add_argument("--end", type=int, default=None, help="End index in the loaded dataset")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for pending samples")
    parser.add_argument("--seed", type=int, default=42, help="Seed forwarded to the model API")
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--few_shot_seed", type=int, default=42)
    parser.add_argument("--few_shot_strategy", choices=["random", "similar"], default="random")
    parser.add_argument("--few_shot_source", type=str, default=None)
    parser.add_argument("--acc_samples", type=int, default=1)
    args = parser.parse_args()

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    existing_results = load_existing_results(args.save_path)
    completed_sample_ids = set(existing_results.keys())

    all_data = load_samples(args.data_dir)
    full_data = all_data[args.start : args.end]
    pending_samples = filter_pending_samples(full_data, completed_sample_ids)
    if args.max_samples is not None:
        pending_samples = pending_samples[: args.max_samples]

    if not pending_samples:
        print("所有样本都已完成评测！")
        return

    results_by_id = dict(existing_results)
    save_lock = threading.Lock()
    generation_kwargs = {
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    prompt_template = load_prompt_template()
    few_shot_pool: List[dict] = []
    if args.few_shot > 0:
        if not args.few_shot_source:
            raise ValueError("few_shot_source is required when few_shot > 0")
        if args.few_shot_source:
            source_paths = [
                source.strip()
                for source in args.few_shot_source.split(",")
                if source.strip()
            ]
            if all(source.endswith(".json") for source in source_paths):
                for source in source_paths:
                    few_shot_pool.extend(
                        load_result_fewshot_pool(
                            source,
                            all_data,
                            lambda item: render_prompt(prompt_template, item),
                            source_label=os.path.basename(os.path.dirname(os.path.dirname(source))) or os.path.basename(source),
                        )
                    )
            else:
                few_shot_pool = load_sft_fewshot_pool("bench4opt_orgeval", args.few_shot_source)
        print(
            f"Loaded {len(few_shot_pool)} Bench4Opt ORGEval few-shot examples "
            f"(strategy={args.few_shot_strategy})"
        )
    completed_since_last_save = 0
    max_workers = max(1, min(args.max_workers, len(pending_samples)))
    eval_workers = max(1, args.eval_workers)
    _eval_semaphore = threading.Semaphore(eval_workers)

    print(
        f"使用快速评测模式: workers={max_workers}, eval_workers={eval_workers}, "
        f"eval_timeout={args.eval_timeout}s, save_every={args.save_every}"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample_id = {
            executor.submit(
                run_sample,
                args.data_dir,
                sample,
                args.model_name,
                prompt_template,
                generation_kwargs,
                args.eval_timeout,
                args.openai_api_key,
                args.openai_base_url,
                few_shot_pool,
                args.few_shot,
                args.few_shot_seed,
                args.few_shot_strategy,
                args.acc_samples,
            ): sample["id"]
            for sample in pending_samples
        }

        for future in tqdm(as_completed(future_to_sample_id), total=len(future_to_sample_id), desc="快速评测"):
            sample_id = future_to_sample_id[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"样本 {sample_id} 处理失败: {exc}")
                continue

            results_by_id[sample_id] = result
            completed_since_last_save += 1

            if completed_since_last_save >= args.save_every:
                save_results_incremental(sort_results_by_numeric_id(results_by_id), args.save_path, save_lock)
                completed_since_last_save = 0

    save_results_incremental(sort_results_by_numeric_id(results_by_id), args.save_path, save_lock)
    print(f"评测完成！总共保存了 {len(results_by_id)} 个样本，结果已保存到 {args.save_path}")


if __name__ == "__main__":
    main()
