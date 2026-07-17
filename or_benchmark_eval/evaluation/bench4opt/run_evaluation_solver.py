import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from openai import OpenAI
from tqdm import tqdm

from evaluation.common.fewshot_acc import (
    attempt_seed,
    augment_chat_messages,
    fewshot_metadata,
    load_result_fewshot_pool,
    load_sft_fewshot_pool,
    select_fewshot_examples,
)
from evaluation.bench4opt.utils.lp_utils import extract_python_code
from evaluation.optibench.solver_evaluation import evaluate_with_solver


PROMPT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "prompt_template.txt"
)

_thread_local = threading.local()


def load_prompt_template() -> str:
    with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as handle:
        return handle.read()


def get_gurobi_version() -> str:
    try:
        import gurobipy as gp

        return ".".join(str(part) for part in gp.gurobi.version())
    except Exception:
        return "unknown"


def render_prompt_messages(
    template: str,
    sample: Dict[str, object],
    data_dir: str,
) -> List[Dict[str, str]]:
    if "SYSTEM:" not in template or "USER:" not in template:
        raise ValueError(f"Prompt template must contain SYSTEM: and USER: sections: {PROMPT_TEMPLATE_PATH}")

    _, remainder = template.split("SYSTEM:", 1)
    system_content, user_content = remainder.split("USER:", 1)

    system_content = system_content.strip().replace("{{GurobiVersion}}", get_gurobi_version())
    user_content = user_content.strip()
    user_content = user_content.replace("{{Problem}}", str(sample["problem"]))
    user_content = user_content.replace("{{DataPath}}", str(sample["data_path"]))
    user_content = user_content.replace("{{WorkingDirectory}}", os.path.abspath(data_dir))

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def load_samples(data_dir: str) -> List[Dict[str, object]]:
    test_path = os.path.join(data_dir, "test.jsonl")
    with open(test_path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def get_openai_client(api_key: str, base_url: str):
    client = getattr(_thread_local, "openai_client", None)
    client_key = getattr(_thread_local, "openai_client_key", None)
    client_base_url = getattr(_thread_local, "openai_client_base_url", None)

    if client is None or client_key != api_key or client_base_url != base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
        _thread_local.openai_client = client
        _thread_local.openai_client_key = api_key
        _thread_local.openai_client_base_url = base_url
    return client


def generate_completion(
    messages: List[Dict[str, str]],
    model_name: str,
    openai_api_key: str,
    openai_base_url: str,
    **generation_kwargs,
) -> str:
    client = get_openai_client(openai_api_key, openai_base_url)

    request_kwargs = {
        "model": model_name,
        "seed": generation_kwargs.get("seed", 42),
        "stop": [],
    }
    request_kwargs.update({k: v for k, v in generation_kwargs.items() if v is not None})

    from openai import APIError, InternalServerError

    for attempt in range(3):
        try:
            request_output = client.chat.completions.create(
                messages=messages,
                **request_kwargs,
            )
            return request_output.choices[0].message.content or ""
        except (InternalServerError, APIError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def estimate_completion_tokens(completion: str) -> int:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(completion)) if completion else 0
    except Exception:
        return len(completion) // 4 if completion else 0


def load_instance_data(data_dir: str, data_path: str) -> Dict[str, object]:
    resolved = os.path.join(data_dir, data_path)
    with open(resolved, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {resolved}, got {type(payload)}")
    return payload


def evaluate_one_sample(
    data_dir: str,
    sample: Dict[str, object],
    model_name: str,
    prompt_template: str,
    generation_kwargs: Dict[str, object],
    timeout: int,
    tolerance: float,
    openai_api_key: str,
    openai_base_url: str,
    verbose: bool,
    few_shot_pool: List[Dict[str, str]] | None = None,
    few_shot: int = 0,
    few_shot_seed: int = 42,
    acc_samples: int = 1,
) -> Dict[str, object]:
    sample_id = str(sample.get("id", sample.get("sample_id", -1)))

    try:
        gt_optimal_value = sample.get("optimal_value")
        if gt_optimal_value is None:
            return {
                "sample_id": sample_id,
                "id": sample_id,
                "success": False,
                "predicted_optimal_value": None,
                "gt_optimal_value": None,
                "error": None,
                "message": "Missing optimal_value in sample",
                "completion": None,
                "extracted_code": "",
                "completion_length": 0,
                "evaluation_time": 0.0,
                "api_error": False,
                "data_path": sample.get("data_path"),
                "reference_lp_path": sample.get("reference_lp_path"),
            }

        messages = render_prompt_messages(prompt_template, sample, data_dir)
        selected_examples = select_fewshot_examples(
            few_shot_pool or [], sample_id, few_shot, few_shot_seed
        )
        messages = augment_chat_messages(messages, selected_examples)
        instance_data = load_instance_data(data_dir, str(sample["data_path"]))

        attempts = []
        requested_attempts = max(1, int(acc_samples))
        base_seed = int(generation_kwargs.get("seed", 42))
        selected_result = None

        for attempt_index in range(requested_attempts):
            attempt_kwargs = dict(generation_kwargs)
            attempt_kwargs["seed"] = attempt_seed(base_seed, attempt_index)
            completion = generate_completion(
                messages,
                model_name,
                openai_api_key,
                openai_base_url,
                **attempt_kwargs,
            )
            extracted_code = extract_python_code(completion)

            eval_start_time = time.time()
            eval_result = evaluate_with_solver(
                code=completion,
                gt_optimal_value=float(gt_optimal_value),
                data=instance_data,
                tolerance=tolerance,
                timeout=timeout,
                verbose=verbose,
            )
            evaluation_time = time.time() - eval_start_time

            attempt_result = {
                "attempt_index": attempt_index,
                "seed": attempt_kwargs["seed"],
                "sample_id": sample_id,
                "id": sample_id,
                "success": eval_result["success"],
                "predicted_optimal_value": eval_result["predicted_optimal_value"],
                "gt_optimal_value": eval_result["gt_optimal_value"],
                "error": eval_result.get("error"),
                "message": eval_result["message"],
                "completion": completion,
                "extracted_code": extracted_code,
                "completion_length": estimate_completion_tokens(completion),
                "evaluation_time": evaluation_time,
                "api_error": False,
                "data_path": sample.get("data_path"),
                "reference_lp_path": sample.get("reference_lp_path"),
            }
            attempts.append(attempt_result)
            if attempt_result["success"]:
                selected_result = attempt_result
                break

        if selected_result is None:
            selected_result = attempts[-1]

        result = {
            key: value
            for key, value in selected_result.items()
            if key not in {"attempt_index", "seed"}
        }
        result["acc_samples_requested"] = requested_attempts
        result["acc_attempts_run"] = len(attempts)
        result["acc_success_attempt"] = selected_result["attempt_index"] if selected_result["success"] else None
        result["attempts"] = attempts
        result.update(fewshot_metadata(selected_examples))
        return result
    except Exception as exc:
        if verbose:
            import traceback

            print(f"Error evaluating sample {sample_id}: {exc}")
            traceback.print_exc()

        return {
            "sample_id": sample_id,
            "id": sample_id,
            "success": False,
            "predicted_optimal_value": None,
            "gt_optimal_value": sample.get("optimal_value"),
            "error": None,
            "message": f"Error: {exc}",
            "completion": None,
            "extracted_code": "",
            "completion_length": 0,
            "evaluation_time": 0.0,
            "api_error": True,
            "data_path": sample.get("data_path"),
            "reference_lp_path": sample.get("reference_lp_path"),
            "acc_samples_requested": max(1, int(acc_samples)),
            "acc_attempts_run": 0,
            "acc_success_attempt": None,
        }


def load_existing_results(save_path: str) -> Dict[str, Dict[str, object]]:
    if not os.path.exists(save_path):
        return {}

    try:
        with open(save_path, "r", encoding="utf-8") as handle:
            results = json.load(handle)
    except Exception as exc:
        print(f"Warning: failed to load existing results from {save_path}: {exc}")
        return {}

    completed_samples = {}
    for result in results:
        sample_id = result.get("id") or result.get("sample_id")
        if sample_id:
            completed_samples[str(sample_id)] = result
    return completed_samples


def save_results_incremental(all_results: List[Dict[str, object]], save_path: str, lock: threading.Lock) -> None:
    with lock:
        tmp_path = save_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(all_results, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, save_path)


def sort_results_by_numeric_id(results_by_id: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    def sample_sort_key(sample_id: str):
        try:
            return int(sample_id.split("_")[1])
        except (IndexError, ValueError):
            return sample_id

    return [results_by_id[sample_id] for sample_id in sorted(results_by_id.keys(), key=sample_sort_key)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Solver evaluation for bench4opt feasible subset")
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/bench4opt_feasible")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--openai_base_url", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--max_workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--save_every", type=int, default=16)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--few_shot_seed", type=int, default=42)
    parser.add_argument("--few_shot_source", type=str, default=None)
    parser.add_argument("--acc_samples", type=int, default=1)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--verbose", type=lambda x: str(x).lower() == "true", default=False)
    args = parser.parse_args()

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    full_data = load_samples(args.data_dir)
    end = len(full_data) if args.end is None else min(args.end, len(full_data))
    selected_samples = full_data[args.start:end]

    existing_results = {} if args.rerun else load_existing_results(args.save_path)
    pending_samples = [sample for sample in selected_samples if str(sample["id"]) not in existing_results]

    if args.max_samples is not None:
        pending_samples = pending_samples[: args.max_samples]

    if not pending_samples and existing_results:
        print("All selected samples already have results.")
        return

    results_by_id = dict(existing_results)
    save_lock = threading.Lock()
    generation_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }
    prompt_template = load_prompt_template()
    few_shot_pool: List[Dict[str, str]] = []
    if args.few_shot > 0:
        if not args.few_shot_source:
            raise ValueError("few_shot_source is required when few_shot > 0")
        if args.few_shot_source:
            if args.few_shot_source.endswith(".json"):
                few_shot_pool = load_result_fewshot_pool(
                    args.few_shot_source,
                    full_data,
                    lambda item: render_prompt_messages(prompt_template, item, args.data_dir)[1]["content"],
                    source_label=os.path.basename(args.few_shot_source),
                )
            else:
                few_shot_pool = load_sft_fewshot_pool("bench4opt_feasible_solver", args.few_shot_source)
        print(f"Loaded {len(few_shot_pool)} Bench4Opt feasible few-shot examples")
    completed_since_last_save = 0
    max_workers = max(1, min(args.max_workers, len(pending_samples))) if pending_samples else 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample_id = {
            executor.submit(
                evaluate_one_sample,
                args.data_dir,
                sample,
                args.model_name,
                prompt_template,
                generation_kwargs,
                args.timeout,
                args.tolerance,
                args.openai_api_key,
                args.openai_base_url,
                args.verbose,
                few_shot_pool,
                args.few_shot,
                args.few_shot_seed,
                args.acc_samples,
            ): str(sample["id"])
            for sample in pending_samples
        }

        for future in tqdm(as_completed(future_to_sample_id), total=len(future_to_sample_id), desc="bench4opt feasible solver"):
            sample_id = future_to_sample_id[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"Sample {sample_id} failed: {exc}")
                continue

            results_by_id[sample_id] = result
            completed_since_last_save += 1

            if args.save_every > 0 and completed_since_last_save >= args.save_every:
                save_results_incremental(sort_results_by_numeric_id(results_by_id), args.save_path, save_lock)
                completed_since_last_save = 0

    save_results_incremental(sort_results_by_numeric_id(results_by_id), args.save_path, save_lock)
    print(f"Saved {len(results_by_id)} results to {args.save_path}")


if __name__ == "__main__":
    main()
