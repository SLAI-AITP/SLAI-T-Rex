# Examples

This directory is the example index for **SLAI T-Rex**.

The current runnable example lives inside the SFT data construction package:

```text
../sft_data_construction/examples/configs/demo.yaml
../sft_data_construction/examples/seeds/small_seed.jsonl
```

Run it without calling an LLM:

```bash
cd SLAI-T-Rex/sft_data_construction
python3 -m or_data_distill run \
  --config examples/configs/demo.yaml \
  --dry-run
```

Future examples should connect the full public workflow:

```text
public OR seed subset
  -> SFT data distillation dry-run
  -> real OpenAI-compatible generation
  -> SFT JSONL validation
  -> MindSpeed data conversion
  -> SFT launcher configuration
  -> HF export and evaluation manifest
```

Large checkpoints, private cluster configs, solver licenses, and production evaluation data should stay outside this repository.
