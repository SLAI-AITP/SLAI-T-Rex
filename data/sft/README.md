# SFT Data Construction

Turns OR problem–answer seeds into OpenAI-style SFT JSONL (IR → render → quality gate).

```bash
cd SLAI-T-Rex/data/sft
python3 -m pip install -e .

python3 -m or_data_distill validate-sft --input seeds/public_seed.jsonl
python3 -m or_data_distill run --config examples/configs/demo.yaml --dry-run
```

Real generation:

```bash
cp configs/run.example.yaml configs/run.local.yaml
# set llm.base_url / model
export LLM_API_KEY=YOUR_KEY_IF_NEEDED
python3 -m or_data_distill run --config configs/run.local.yaml
python3 -m or_data_distill validate-sft --input runs/sft_data_demo/sft.jsonl
```

Accepted rows: `runs/<run_id>/sft.jsonl`. Edit `configs/run.example.yaml` for quotas, multi-endpoint `llm.base_urls`, and flywheel `synthetic_pool`.
