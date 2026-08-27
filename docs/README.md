# Documentation

This directory is the documentation index for **SLAI T-Rex**.

The root README gives the public quick start. Module-level READMEs document runnable commands. This directory holds longer notes and the technical report PDF.

Current documents:

- [SLAI-T-Rex.pdf](SLAI-T-Rex.pdf) for the technical report;
- [../data/cpt/README.md](../data/cpt/README.md) for the OR-CPT engine;

The current runnable example lives inside the SFT data construction package:

```text
../data/sft/examples/configs/demo.yaml
../data/sft/examples/seeds/small_seed.jsonl
```

Run it without calling an LLM:

```bash
cd SLAI-T-Rex/data/sft
python3 -m or_data_distill run \
  --config examples/configs/demo.yaml \
  --dry-run
```

For now, use:

- [../README.md](../README.md) for the project overview;
- [../data/cpt/README.md](../data/cpt/README.md) for the OR-CPT engine;
- [../data/sft/README.md](../data/sft/README.md) for runnable SFT data construction;
- [../training/cpt/README.md](../training/cpt/README.md) and [../training/sft/README.md](../training/sft/README.md) for training templates;
- [../eval/README.md](../eval/README.md) for OR benchmark evaluation.
