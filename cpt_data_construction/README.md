# CPT Data Construction

This module records the public release scope for the OR-CPT data construction stage of **SLAI T-Rex**.

In the technical report, OR-CPT data is built through a solver-verified synthesis engine: parameterized optimization generators create structured instances, Gurobi verifies feasibility and objective values, business-oriented problem statements are rendered, executable formulations are reconstructed, and contract checks decide whether a document is eligible for CPT.

## Current Release

This directory is a design placeholder. The production OR-CPT engine, private raw corpora, solver logs, and generated full-scale CPT documents are not included in this repository release.

The released repository instead provides:

- the technical report PDF and LaTeX source archive at the repository root;
- the runnable SFT data distillation toolkit in `../sft_data_construction`;
- CPT conversion and training launch templates in `../cpt_training`.

## Target Pipeline

```text
OR source resources
  -> generator contracts
  -> parameterized optimization instance
  -> solver execution and objective verification
  -> business problem rendering
  -> executable formulation reconstruction
  -> contract checks
  -> solver-verified CPT document
```

## Release Requirements

Future public releases of this module should include:

- generator contracts and schema definitions;
- static validation rules for variables, constraints, units, and objective direction;
- independent solver execution scripts;
- objective-matching and feasibility checks;
- provenance metadata from source resource to exported CPT document;
- dataset cards that separate public, synthetic, and private-source components.

For runnable SFT data generation today, start from [sft_data_construction](../sft_data_construction/).
