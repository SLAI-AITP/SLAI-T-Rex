"""基于 solver 的 OptiBench 评测脚本。"""

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import HfArgumentParser

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

OBJECTIVE_KEY_PATTERNS = [
    re.compile(
        r"(optimal|maximum|minimum|maximized|minimized|max|min|profit|cost|"
        r"revenue|objective|ratio|value|area|volume|perimeter|surface|time|"
        r"distance|earnings|income|wage|npv|exposure|audience|utility|output|"
        r"pollution|sodium|sugar|fat|calories?|radiation|light|donations|"
        r"payout|energy|total)",
        re.IGNORECASE,
    ),
    re.compile(
        r"amount of .* (used|required|needed|produced|processed|transported|delivered|"
        r"picked|extracted|consumed|converted)",
        re.IGNORECASE,
    ),
    re.compile(
        r"number of .* (delivered|processed|transported|stored|treated|performed)",
        re.IGNORECASE,
    ),
    re.compile(r"seating availability", re.IGNORECASE),
    re.compile(r"product of the two numbers", re.IGNORECASE),
    re.compile(r"length of the rope required", re.IGNORECASE),
    re.compile(r"mass that can be supported", re.IGNORECASE),
]

QUESTION_TARGET_PATTERNS = [
    re.compile(r"minimize the total number of ([a-z\- ]+?)(?: needed| used| required)?$", re.IGNORECASE),
    re.compile(r"reduce the total number of ([a-z\- ]+?)(?: needed| used| required)?$", re.IGNORECASE),
    re.compile(r"decrease the total number of ([a-z\- ]+?)(?: needed| used| required)?$", re.IGNORECASE),
    re.compile(r"find the minimum number of ([a-z\- ]+?)(?: that can be used)?$", re.IGNORECASE),
    re.compile(r"minimize the total number of ([a-z\- ]+?)$", re.IGNORECASE),
]

COUNT_LIKE_KEY_PATTERN = re.compile(
    r"^(the )?(number|quantity|hours|workers|acres|amount invested|investment)",
    re.IGNORECASE,
)
FERTILIZER_KEY_PATTERN = re.compile(r"^The quantity of Fertilizer \d+$")
FERTILIZER_PRICE_PATTERN = re.compile(r"\$([0-9]+(?:\.[0-9]+)?)\s*per\s*100\s*pounds", re.IGNORECASE)
FERTILIZER_TABLE_PRICE_PATTERN = re.compile(
    r"\|\s*Fertilizer\s+\d+\s*\|[^\n]*?\|\s*([0-9]+(?:\.[0-9]+)?)\s*(?:\n|\|)",
    re.IGNORECASE,
)


@dataclass
class Arguments:
    dataset_name: str = field(
        default="./data/optibench",
        metadata={"help": "The dataset path or dataset name to use"},
    )
    split: str = field(default="test", metadata={"help": "The split to use"})

    model_name: str = field(
        default="gpt-4o", metadata={"help": "The model name to use"}
    )
    openai_api_key: str = field(
        default=None, metadata={"help": "The API key for the model service"}
    )
    openai_base_url: str = field(
        default=None,
        metadata={"help": "The base URL for the model service"},
    )

    temperature: float = field(
        default=0.0, metadata={"help": "The temperature for the model"}
    )
    top_p: float = field(default=1.0, metadata={"help": "The top p for the model"})
    max_tokens: int = field(
        default=8192, metadata={"help": "The max tokens for the model"}
    )
    seed: int = field(default=42, metadata={"help": "The seed for the model"})

    batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "The batch size for processing. If None, will be set to num_workers"
        },
    )
    num_workers: int = field(
        default=8,
        metadata={"help": "Number of concurrent workers for parallel processing"},
    )
    start: int = field(default=0, metadata={"help": "The start index"})
    end: Optional[int] = field(
        default=None, metadata={"help": "The end index"}
    )
    shuffle_dataset: bool = field(
        default=False,
        metadata={"help": "Whether to shuffle the dataset before processing"},
    )

    timeout: float = field(
        default=360.0,
        metadata={"help": "Time limit in seconds for solver execution"},
    )
    tolerance: float = field(
        default=1e-6,
        metadata={"help": "Tolerance for comparing optimal values"},
    )

    save_path: Optional[str] = field(
        default="solver_evaluation.json",
        metadata={"help": "The path to save evaluation results"},
    )
    rerun: bool = field(
        default=False,
        metadata={
            "help": "Whether to rerun evaluation even if some results already exist"
        },
    )
    verbose: bool = field(
        default=False, metadata={"help": "Whether to print verbose information"}
    )
    few_shot: int = field(
        default=0,
        metadata={"help": "Number of solved examples to include before each current question"},
    )
    few_shot_seed: int = field(
        default=42,
        metadata={"help": "Seed for deterministic few-shot example selection"},
    )
    few_shot_source: Optional[str] = field(
        default=None,
        metadata={"help": "Validated messages jsonl used as full-answer few-shot pool"},
    )
    acc_samples: int = field(
        default=1,
        metadata={"help": "Sample each question this many times; any successful attempt marks the item correct"},
    )


def load_prompt_template() -> str:
    with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as handle:
        return handle.read()


def get_gurobi_version() -> str:
    try:
        import gurobipy as gp

        return ".".join(str(part) for part in gp.gurobi.version())
    except Exception:
        return "unknown"


def extract_question_text(sample: Dict[str, Any]) -> str:
    if "question" in sample:
        return str(sample["question"])
    if "en_question" in sample:
        return str(sample["en_question"])
    if "prompt" in sample:
        prompt = sample["prompt"]
        if isinstance(prompt, list):
            for message in reversed(prompt):
                if isinstance(message, dict) and message.get("role") == "user" and message.get("content"):
                    return str(message["content"])
            return "\n\n".join(
                str(message.get("content", "")) if isinstance(message, dict) else str(message)
                for message in prompt
            )
        return str(prompt)
    raise ValueError(f"Cannot find prompt/question in sample. Available keys: {list(sample.keys())}")


def render_prompt_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    template = load_prompt_template()
    if "SYSTEM:" not in template or "USER:" not in template:
        raise ValueError(f"Prompt template must contain SYSTEM: and USER: sections: {PROMPT_TEMPLATE_PATH}")

    _, remainder = template.split("SYSTEM:", 1)
    system_content, user_content = remainder.split("USER:", 1)

    system_content = system_content.strip().replace("{{GurobiVersion}}", get_gurobi_version())
    user_content = user_content.strip().replace("{{Question}}", extract_question_text(sample))

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def parse_numeric_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        return float(cleaned)
    raise ValueError(f"Unsupported numeric value type: {type(value)}")


def load_results_items(sample: Dict[str, Any]) -> List[Tuple[str, float]]:
    results = sample.get("results", {})
    if isinstance(results, str):
        results = json.loads(results)
    if not isinstance(results, dict):
        raise ValueError(f"Unsupported results type: {type(results)}")

    items: List[Tuple[str, float]] = []
    for key, value in results.items():
        if value is None:
            continue
        items.append((key, parse_numeric_value(value)))
    return items


def find_explicit_objective(items: List[Tuple[str, float]]) -> Tuple[Optional[float], Optional[str]]:
    for key, value in reversed(items):
        if any(pattern.search(key) for pattern in OBJECTIVE_KEY_PATTERNS):
            return value, key
    return None, None


def is_count_like_key(key: str) -> bool:
    return bool(COUNT_LIKE_KEY_PATTERN.search(key))


def derive_from_question_target(question: str, items: List[Tuple[str, float]]) -> Tuple[Optional[float], Optional[str]]:
    normalized_question = normalize_text(question)

    for pattern in QUESTION_TARGET_PATTERNS:
        match = pattern.search(normalized_question)
        if not match:
            continue

        target_phrase = match.group(1).strip()
        candidate_items = [
            (key, value)
            for key, value in items
            if target_phrase and target_phrase in normalize_text(key)
        ]

        if len(candidate_items) == 1:
            return candidate_items[0][1], candidate_items[0][0]

        if len(candidate_items) > 1 and all(is_count_like_key(key) for key, _ in candidate_items):
            return sum(value for _, value in candidate_items), f"sum({target_phrase})"

        if all(is_count_like_key(key) for key, _ in items):
            return sum(value for _, value in items), "sum(all_count_like_results)"

    return None, None


def derive_open_top_box_surface_area(question: str, items: List[Tuple[str, float]]) -> Tuple[Optional[float], Optional[str]]:
    normalized_question = normalize_text(question)
    if "square base" not in normalized_question or "surface area" not in normalized_question:
        return None, None
    if "open top" not in normalized_question and "open top is to be constructed" not in normalized_question:
        return None, None

    side = None
    height = None
    for key, value in items:
        normalized_key = normalize_text(key)
        if "side of the square base" in normalized_key:
            side = value
        elif normalized_key.endswith("height of the box"):
            height = value

    if side is None or height is None:
        return None, None

    return side * side + 4.0 * side * height, "derived_open_top_box_surface_area"


def derive_fertilizer_cost(question: str, items: List[Tuple[str, float]]) -> Tuple[Optional[float], Optional[str]]:
    if len(items) != 5:
        return None, None
    if not all(FERTILIZER_KEY_PATTERN.match(key) for key, _ in items):
        return None, None

    prices = [float(price) for price in FERTILIZER_PRICE_PATTERN.findall(question)]
    if len(prices) < 5:
        prices = [float(price) for price in FERTILIZER_TABLE_PRICE_PATTERN.findall(question)]
    if len(prices) < 5:
        return None, None

    gt_value = sum(value * price for (_, value), price in zip(items, prices[:5]))
    return gt_value, "derived_fertilizer_weighted_cost"


def extract_ground_truth_value(sample: Dict[str, Any]) -> Tuple[Optional[float], str, Optional[str]]:
    if sample.get("en_answer") is not None:
        try:
            return parse_numeric_value(sample["en_answer"]), "Using en_answer field", "en_answer"
        except Exception as exc:
            return None, f"Could not parse en_answer: {exc}", "en_answer"

    try:
        items = load_results_items(sample)
    except Exception as exc:
        return None, f"Could not parse results: {exc}", None

    if not items:
        return None, "No results found in sample", None

    if len(items) == 1:
        key, value = items[0]
        return value, f"Using single result field: {key}", key

    explicit_value, explicit_key = find_explicit_objective(items)
    if explicit_value is not None:
        return explicit_value, f"Using objective-like result field: {explicit_key}", explicit_key

    question = sample.get("question", "")

    derived_value, derived_key = derive_from_question_target(question, items)
    if derived_value is not None:
        return derived_value, f"Derived target from question: {derived_key}", derived_key

    derived_value, derived_key = derive_open_top_box_surface_area(question, items)
    if derived_value is not None:
        return derived_value, f"Derived target from geometry: {derived_key}", derived_key

    derived_value, derived_key = derive_fertilizer_cost(question, items)
    if derived_value is not None:
        return derived_value, f"Derived target from fertilizer prices: {derived_key}", derived_key

    return None, "Could not derive a scalar ground-truth objective from sample results", None


def resolve_local_data_file(dataset_name: str, split: str) -> Optional[str]:
    if os.path.isfile(dataset_name):
        return os.path.abspath(dataset_name)

    if not os.path.isdir(dataset_name):
        return None

    candidates = [
        os.path.join(dataset_name, "OptiBench.json"),
        os.path.join(dataset_name, f"{split}.json"),
        os.path.join(dataset_name, f"{split}.jsonl"),
        os.path.join(dataset_name, f"{split}.parquet"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def load_local_optibench_dataset(local_data_file: str) -> Dataset:
    extension = os.path.splitext(local_data_file)[1].lower()
    if extension == ".json":
        with open(local_data_file, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        if isinstance(records, dict):
            records = records.get("data", records.get("records", records))
        if not isinstance(records, list):
            raise ValueError(f"Unsupported JSON dataset shape in {local_data_file}")
        return Dataset.from_list(records)

    if extension == ".jsonl":
        records = []
        with open(local_data_file, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return Dataset.from_list(records)

    if extension == ".parquet":
        return load_dataset("parquet", data_files={"test": local_data_file}, split="test")

    raise ValueError(f"Unsupported local dataset extension: {local_data_file}")


def load_optibench_dataset(dataset_name: str, split: str):
    local_data_file = resolve_local_data_file(dataset_name, split)
    if local_data_file is None:
        return load_dataset(dataset_name, split=split)

    return load_local_optibench_dataset(local_data_file)


def generate_completions_with_openai_api(client, message, model_name, **kwargs) -> List[str]:
    from openai import APIError, InternalServerError

    sampling_kwargs = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 32000,
        "n": 1,
        "seed": 42,
    }
    sampling_kwargs.update(kwargs)

    if "stop" not in kwargs or kwargs.get("stop") is None:
        sampling_kwargs["stop"] = []

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=message,
                **sampling_kwargs,
            )
            return [choice.message.content or "" for choice in response.choices]
        except (InternalServerError, APIError):
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def build_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    return render_prompt_messages(sample)


def build_sample_data(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "data" in sample:
        data = sample["data"]
        if isinstance(data, str):
            data = json.loads(data)
        return data
    if "question" in sample:
        return {"question": sample["question"]}
    if "en_question" in sample:
        return {"question": sample["en_question"]}
    return None


def estimate_completion_tokens(completion: str) -> int:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(completion)) if completion else 0
    except Exception:
        return len(completion) // 4 if completion else 0


def evaluate_one_sample(
    sample: Dict[str, Any],
    client,
    model_name: str,
    generation_kwargs: Dict[str, Any],
    timeout: float,
    tolerance: float,
    verbose: bool = False,
    few_shot_pool: Optional[List[Dict[str, str]]] = None,
    few_shot: int = 0,
    few_shot_seed: int = 42,
    acc_samples: int = 1,
) -> Dict[str, Any]:
    sample_id = sample.get("sample_id", sample.get("index", sample.get("id", -1)))

    try:
        gt_optimal_value, gt_message, gt_source = extract_ground_truth_value(sample)
        if gt_optimal_value is None:
            return {
                "sample_id": sample_id,
                "success": False,
                "predicted_optimal_value": None,
                "gt_optimal_value": None,
                "error": None,
                "message": gt_message,
                "completion": None,
                "extracted_code": "",
                "completion_length": 0,
                "evaluation_time": 0.0,
                "api_error": False,
                "unsupported_ground_truth": True,
                "gt_source": gt_source,
            }

        messages = build_messages(sample)
        selected_examples = select_fewshot_examples(
            few_shot_pool or [], sample_id, few_shot, few_shot_seed
        )
        messages = augment_chat_messages(messages, selected_examples)

        data = build_sample_data(sample)

        attempts = []
        requested_attempts = max(1, int(acc_samples))
        base_seed = int(generation_kwargs.get("seed", 42))
        selected_result = None

        for attempt_index in range(requested_attempts):
            attempt_kwargs = dict(generation_kwargs)
            attempt_kwargs["seed"] = attempt_seed(base_seed, attempt_index)
            completions = generate_completions_with_openai_api(
                client,
                messages,
                model_name,
                **attempt_kwargs,
            )
            completion = completions[0] if completions else ""
            extracted_code = extract_python_code(completion)

            eval_start_time = time.time()
            eval_result = evaluate_with_solver(
                code=completion,
                gt_optimal_value=gt_optimal_value,
                data=data,
                tolerance=tolerance,
                timeout=int(timeout),
                verbose=verbose,
            )
            evaluation_time = time.time() - eval_start_time

            attempt_result = {
                "attempt_index": attempt_index,
                "seed": attempt_kwargs["seed"],
                "success": eval_result["success"],
                "predicted_optimal_value": eval_result["predicted_optimal_value"],
                "gt_optimal_value": eval_result["gt_optimal_value"],
                "error": eval_result.get("error"),
                "message": f"{gt_message}; {eval_result['message']}",
                "completion": completion,
                "extracted_code": extracted_code,
                "completion_length": estimate_completion_tokens(completion),
                "evaluation_time": evaluation_time,
                "api_error": False,
                "unsupported_ground_truth": False,
                "gt_source": gt_source,
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
        result["sample_id"] = sample_id
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
            "success": False,
            "predicted_optimal_value": None,
            "gt_optimal_value": None,
            "error": None,
            "message": f"Error: {exc}",
            "completion": None,
            "extracted_code": "",
            "completion_length": 0,
            "evaluation_time": 0.0,
            "api_error": True,
            "unsupported_ground_truth": False,
            "gt_source": None,
            "acc_samples_requested": max(1, int(acc_samples)),
            "acc_attempts_run": 0,
            "acc_success_attempt": None,
        }


def main(args: Arguments):
    log_args = args.__dict__.copy()
    if log_args.get("openai_api_key"):
        log_args["openai_api_key"] = "***masked***"
    pprint(log_args)

    dataset = load_optibench_dataset(args.dataset_name, args.split)
    if args.end is None:
        args.end = len(dataset)

    if "sample_id" not in dataset.column_names:
        if "index" in dataset.column_names:
            dataset = dataset.add_column("sample_id", list(dataset["index"]))
        else:
            dataset = dataset.add_column("sample_id", list(range(len(dataset))))
    all_dataset = dataset

    dataset = dataset.select(range(args.start, args.end))

    if args.shuffle_dataset:
        import random

        random.seed(args.seed)
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        dataset = dataset.select(indices)
        print(f"Dataset shuffled with seed {args.seed}")

    client = OpenAI(api_key=args.openai_api_key, base_url=args.openai_base_url)

    few_shot_pool = []
    if args.few_shot > 0:
        if not args.few_shot_source:
            raise ValueError("few_shot_source is required when few_shot > 0")
        if args.few_shot_source and args.few_shot_source.endswith(".jsonl"):
            few_shot_pool = load_sft_fewshot_pool("optibench_solver", args.few_shot_source)
        else:
            result_source = args.few_shot_source
            few_shot_pool = load_result_fewshot_pool(
                result_source,
                list(all_dataset),
                lambda item: render_prompt_messages(item)[-1]["content"],
                source_label=str(result_source),
            )
        print(f"Loaded {len(few_shot_pool)} OptiBench few-shot examples")
    generation_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }

    if args.batch_size is None:
        args.batch_size = args.num_workers
        print(f"batch_size set to {args.batch_size} (same as num_workers)")

    output_data: List[Dict[str, Any]] = []
    completed_sample_ids = set()

    if args.save_path and os.path.exists(args.save_path):
        try:
            with open(args.save_path, "r") as file:
                if args.save_path.endswith(".json"):
                    output_data = json.load(file)
                elif args.save_path.endswith(".jsonl"):
                    for line in file:
                        output_data.append(json.loads(line))
                else:
                    raise ValueError(f"Unsupported file extension: {args.save_path}")

            api_error_sample_ids = set()
            for item in output_data:
                if "sample_id" not in item:
                    continue
                if item.get("api_error", False):
                    api_error_sample_ids.add(item["sample_id"])
                    continue
                if "success" in item or item.get("unsupported_ground_truth", False):
                    completed_sample_ids.add(item["sample_id"])

            print(f"Loaded {len(output_data)} existing results from {args.save_path}")
            print(f"Found {len(completed_sample_ids)} completed samples")
            if api_error_sample_ids:
                print(
                    f"Found {len(api_error_sample_ids)} samples with API errors that will be retried"
                )
        except Exception as exc:
            print(f"Warning: Could not load existing results from {args.save_path}: {exc}")
            output_data = []
            completed_sample_ids = set()

    if args.rerun:
        output_data = []
        completed_sample_ids = set()
        data_to_process = dataset
        print("Rerun mode: will process all samples and ignore existing saved results")
    else:
        incomplete_indices = [
            index
            for index, sample in enumerate(dataset)
            if sample["sample_id"] not in completed_sample_ids
        ]
        data_to_process = dataset.select(incomplete_indices) if incomplete_indices else dataset.select([])
        print(f"Found {len(data_to_process)} samples to process (out of {len(dataset)} total)")

    results: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def _worker(sample, **kwargs):
        result = evaluate_one_sample(sample, **kwargs)
        with lock:
            results.append(result)
            if args.save_path:
                directory = os.path.dirname(args.save_path) or "."
                os.makedirs(directory, exist_ok=True)
                with open(args.save_path, "w") as file:
                    json.dump(output_data + results, file, indent=2)
        return result

    if len(data_to_process) > 0:
        samples_list = [sample for sample in data_to_process]
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(
                    _worker,
                    sample,
                    client=client,
                    model_name=args.model_name,
                    generation_kwargs=generation_kwargs,
                    timeout=args.timeout,
                    tolerance=args.tolerance,
                    verbose=args.verbose,
                    few_shot_pool=few_shot_pool,
                    few_shot=args.few_shot,
                    few_shot_seed=args.few_shot_seed,
                    acc_samples=args.acc_samples,
                )
                for sample in samples_list
            ]

            for future in tqdm(futures, desc="Evaluating"):
                try:
                    future.result()
                except Exception as exc:
                    if args.verbose:
                        print(f"Error in worker: {exc}")

    final_results = output_data + results

    if args.save_path:
        directory = os.path.dirname(args.save_path) or "."
        os.makedirs(directory, exist_ok=True)
        with open(args.save_path, "w") as file:
            json.dump(final_results, file, indent=2)
        print(f"Results saved to {args.save_path}")

    total = len(final_results)
    unsupported = sum(1 for result in final_results if result.get("unsupported_ground_truth", False))
    evaluated = total - unsupported
    successful = sum(
        1 for result in final_results if result.get("success", False) and not result.get("unsupported_ground_truth", False)
    )
    api_errors = sum(1 for result in final_results if result.get("api_error", False))
    accuracy = successful / evaluated if evaluated > 0 else 0.0

    print(f"\n{'=' * 50}")
    print("Evaluation Statistics:")
    print(f"{'=' * 50}")
    print(f"Total samples: {total}")
    print(f"Evaluated samples: {evaluated}")
    print(f"Unsupported GT samples: {unsupported}")
    print(f"Successful: {successful}")
    print(f"API errors: {api_errors}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    parsed_args = parser.parse_args_into_dataclasses()[0]
    main(parsed_args)
