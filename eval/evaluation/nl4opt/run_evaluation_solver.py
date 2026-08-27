"""
基于solver的评估脚本：使用gurobi计算optimal value并与ground truth对比
"""
import os
import json
import time
from tqdm import tqdm
from pprint import pprint
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import threading
from transformers import HfArgumentParser, AutoTokenizer
from datasets import load_dataset
from openai import OpenAI

from evaluation.common.fewshot_acc import (
    attempt_seed,
    augment_chat_messages,
    fewshot_metadata,
    load_result_fewshot_pool,
    load_sft_fewshot_pool,
    select_fewshot_examples,
)
from evaluation.nl4opt.solver_evaluation import evaluate_with_solver


PROMPT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "prompt_template.txt"
)


@dataclass
class Arguments:
    dataset_name: str = field(
        default="./data/NL4OPT", metadata={"help": "The name of the dataset to use"}
    )
    split: str = field(default="test", metadata={"help": "The split to use"})

    model_name: str = field(
        default="gpt-4o", metadata={"help": "The name of the model to use"}
    )

    openai_api_key: str = field(
        default=None, metadata={"help": "The api key for openai"}
    )
    openai_base_url: str = field(
        default=None, metadata={"help": "The base url for openai"}
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
        default=None, metadata={"help": "The batch size for processing. If None, will be set to num_workers for optimal parallelization"}
    )
    num_workers: int = field(
        default=8, metadata={"help": "Number of concurrent workers for parallel processing"}
    )
    start: int = field(default=0, metadata={"help": "The start index for the model"})
    end: Optional[int] = field(
        default=None, metadata={"help": "The end index for the model"}
    )

    shuffle_dataset: bool = field(
        default=False,
        metadata={"help": "Whether to shuffle the dataset before processing"},
    )

    timeout: float = field(
        default=360.0,
        metadata={"help": "Time limit in seconds for solver execution (default: 360.0)"}
    )
    
    tolerance: float = field(
        default=1e-6,
        metadata={"help": "Tolerance for comparing optimal values (default: 1e-6)"}
    )

    save_path: Optional[str] = field(
        default="solver_evaluation.json",
        metadata={"help": "The path to save the evaluation results"},
    )
    rerun: bool = field(
        default=False,
        metadata={
            "help": "Whether to rerun the evaluation even if the results already exist"
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


def generate_completions_with_openai_api(
    client, message, model_name, **kwargs
) -> List[str]:
    from openai import InternalServerError, APIError
    
    sampling_kwargs = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 32000,
        "n": 1,
        "seed": 42,
    }
    sampling_kwargs.update(kwargs)
    
    # For long-cot models, we need to prevent early stopping on thinking tags
    if "stop" not in kwargs:
        sampling_kwargs["stop"] = []
    elif kwargs.get("stop") is None:
        sampling_kwargs["stop"] = []
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=message,
                **sampling_kwargs
            )
            return [choice.message.content for choice in response.choices]
        except (InternalServerError, APIError) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                raise


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
    """
    评估单个样本
    """
    sample_id = sample.get("sample_id", sample.get("id", -1))
    
    try:
        messages = render_prompt_messages(sample)
        selected_examples = select_fewshot_examples(
            few_shot_pool or [], sample_id, few_shot, few_shot_seed
        )
        messages = augment_chat_messages(messages, selected_examples)

        # 获取ground truth数据
        # NL4OPT数据集可能没有单独的data字段，代码需要从问题描述中提取信息
        # 这里我们提供一个空的data字典，或者包含question的字典
        data = {}
        if "data" in sample:
            data = sample["data"]
            if isinstance(data, str):
                import json
                data = json.loads(data)
        elif "ground_truth" in sample and "data" in sample["ground_truth"]:
            data = sample["ground_truth"]["data"]
            if isinstance(data, str):
                import json
                data = json.loads(data)
        # NL4OPT数据集没有data字段，提供一个包含question的字典
        elif "en_question" in sample:
            # 对于NL4OPT，代码应该从问题描述中提取数据
            # 这里提供一个最小化的数据结构
            data = {"question": sample.get("en_question", "")}
        else:
            # 如果没有data字段，使用空字典
            data = {}
        
        # 获取ground truth optimal value
        gt_optimal_value = None
        if "optimal_value" in sample:
            gt_optimal_value = sample["optimal_value"]
        elif "ground_truth" in sample and "optimal_value" in sample["ground_truth"]:
            gt_optimal_value = sample["ground_truth"]["optimal_value"]
        elif "en_answer" in sample:  # NL4OPT数据集格式
            gt_optimal_value = float(sample["en_answer"])
        elif "answer" in sample:
            try:
                gt_optimal_value = float(sample["answer"])
            except (ValueError, TypeError):
                gt_optimal_value = sample["answer"]
        else:
            raise ValueError(f"Cannot find optimal_value/answer in sample. Available keys: {list(sample.keys())}")

        from evaluation.bench4opt.utils.lp_utils import extract_python_code

        attempts = []
        requested_attempts = max(1, int(acc_samples))
        base_seed = int(generation_kwargs.get("seed", 42))
        selected_result = None

        for attempt_index in range(requested_attempts):
            attempt_kwargs = dict(generation_kwargs)
            attempt_kwargs["seed"] = attempt_seed(base_seed, attempt_index)

            completions = generate_completions_with_openai_api(
                client, messages, model_name, **attempt_kwargs
            )
            completion = completions[0] if completions else ""

            eval_start_time = time.time()
            eval_result = evaluate_with_solver(
                code=completion,
                gt_optimal_value=gt_optimal_value,
                data=data if data else None,  # 如果data为空字典，传递None
                tolerance=tolerance,
                timeout=int(timeout),
                verbose=verbose,
            )
            evaluation_time = time.time() - eval_start_time

            extracted_code = extract_python_code(completion)

            try:
                import tiktoken

                encoding = tiktoken.get_encoding("cl100k_base")
                completion_tokens = len(encoding.encode(completion)) if completion else 0
            except (ImportError, Exception):
                completion_tokens = len(completion) // 4 if completion else 0

            attempt_result = {
                "attempt_index": attempt_index,
                "seed": attempt_kwargs["seed"],
                "success": eval_result["success"],
                "predicted_optimal_value": eval_result["predicted_optimal_value"],
                "gt_optimal_value": eval_result["gt_optimal_value"],
                "error": eval_result.get("error"),
                "message": eval_result["message"],
                "completion": completion,
                "extracted_code": extracted_code,
                "completion_length": completion_tokens,
                "evaluation_time": evaluation_time,
                "api_error": False,
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
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        if verbose:
            print(f"Error evaluating sample {sample_id}: {error_msg}")
            traceback.print_exc()
        return {
            "sample_id": sample_id,
            "success": False,
            "predicted_optimal_value": None,
            "gt_optimal_value": None,
            "error": None,
            "message": f"Error: {error_msg}",
            "completion": None,
            "extracted_code": "",
            "completion_length": 0,
            "evaluation_time": 0.0,
            "api_error": True,
            "acc_samples_requested": max(1, int(acc_samples)),
            "acc_attempts_run": 0,
            "acc_success_attempt": None,
        }


def main(args):
    log_args = args.__dict__.copy()
    if log_args.get("openai_api_key"):
        log_args["openai_api_key"] = "***masked***"
    pprint(log_args)

    #########################
    # Load data
    #########################

    dataset = load_dataset(args.dataset_name, split=args.split)
    if args.end is None:
        args.end = len(dataset)
    if "sample_id" not in dataset.column_names:
        dataset = dataset.add_column("sample_id", list(range(len(dataset))))
    all_dataset = dataset

    # Select the range of data
    dataset = dataset.select(range(args.start, args.end))

    # Shuffle the dataset if requested
    if args.shuffle_dataset:
        import random

        random.seed(args.seed)
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        dataset = dataset.select(indices)
        print(f"Dataset shuffled with seed {args.seed}")

    #########################
    # Load model client
    #########################

    client = OpenAI(api_key=args.openai_api_key, base_url=args.openai_base_url)
    few_shot_pool = []
    if args.few_shot > 0:
        if not args.few_shot_source:
            raise ValueError("few_shot_source is required when few_shot > 0")
        if args.few_shot_source and args.few_shot_source.endswith(".jsonl"):
            few_shot_pool = load_sft_fewshot_pool("nl4opt_solver", args.few_shot_source)
        else:
            result_source = args.few_shot_source
            few_shot_pool = load_result_fewshot_pool(
                result_source,
                list(all_dataset),
                lambda item: render_prompt_messages(item)[-1]["content"],
                source_label=str(result_source),
            )
        print(f"Loaded {len(few_shot_pool)} NL4OPT few-shot examples")

    generation_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }

    # Auto-set batch_size to num_workers if not specified
    if args.batch_size is None:
        args.batch_size = args.num_workers
        print(f"batch_size set to {args.batch_size} (same as num_workers)")

    #########################
    # Check if the save_path exists and load existing results
    #########################

    output_data = []
    completed_sample_ids = set()
    
    if args.save_path and os.path.exists(args.save_path):
        try:
            with open(args.save_path, "r") as f:
                if args.save_path.endswith(".json"):
                    output_data = json.load(f)
                elif args.save_path.endswith(".jsonl"):
                    for line in f:
                        output_data.append(json.loads(line))
                else:
                    raise ValueError(f"Unsupported file extension: {args.save_path}")
            
            # Extract completed sample IDs
            api_error_sample_ids = set()
            for item in output_data:
                if "sample_id" in item:
                    if item.get("api_error", False):
                        api_error_sample_ids.add(item["sample_id"])
                        continue
                    if "success" in item:
                        completed_sample_ids.add(item["sample_id"])
            
            print(f"Loaded {len(output_data)} existing results from {args.save_path}")
            print(f"Found {len(completed_sample_ids)} completed samples")
            if api_error_sample_ids:
                print(f"Found {len(api_error_sample_ids)} samples with API errors that will be retried")
        except Exception as e:
            print(f"Warning: Could not load existing results from {args.save_path}: {e}")
            output_data = []
            completed_sample_ids = set()

    #########################
    # Filter dataset to only process incomplete samples
    #########################

    if args.rerun:
        data_to_process = dataset
        print("Rerun mode: will process all samples")
    else:
        incomplete_indices = [
            i for i, sample in enumerate(dataset)
            if sample["sample_id"] not in completed_sample_ids
        ]
        data_to_process = dataset.select(incomplete_indices) if incomplete_indices else dataset.select([])
        print(f"Found {len(data_to_process)} samples to process (out of {len(dataset)} total)")

    #########################
    # Generate completions and evaluate
    #########################

    results = []
    lock = threading.Lock()
    
    def _worker(sample, **kwargs):
        result = evaluate_one_sample(sample, **kwargs)
        with lock:
            results.append(result)
            # Save incrementally
            if args.save_path:
                os.makedirs(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else ".", exist_ok=True)
                with open(args.save_path, "w") as f:
                    json.dump(output_data + results, f, indent=2)
        return result
    
    # Process samples
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
                except Exception as e:
                    if args.verbose:
                        print(f"Error in worker: {e}")
    
    #########################
    # Save final results and compute statistics
    #########################

    final_results = output_data + results
    
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else ".", exist_ok=True)
        with open(args.save_path, "w") as f:
            json.dump(final_results, f, indent=2)
        print(f"Results saved to {args.save_path}")
    
    # Compute statistics
    total = len(final_results)
    successful = sum(1 for r in final_results if r.get("success", False))
    accuracy = successful / total if total > 0 else 0.0
    
    print(f"\n{'='*50}")
    print(f"Evaluation Statistics:")
    print(f"{'='*50}")
    print(f"Total samples: {total}")
    print(f"Successful: {successful}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
