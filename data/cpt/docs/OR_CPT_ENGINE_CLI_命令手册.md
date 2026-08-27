# OR-CPT Engine CLI 命令手册

最后更新：2026-06-08  
适用范围：当前仓库 `or_cpt_engine/cli/main.py` 中的 Typer CLI。  
相关说明：系统原理、阶段职责和配置含义请看 `OR_CPT_ENGINE_使用说明.md`；项目摸底和汇报材料请看 `OR_CPT_ENGINE_项目摸底报告.md`。

## 0. 使用约定

所有命令都可以用模块方式运行：

```bash
python -m or_cpt_engine.cli.main <command> [options]
```

如果本地安装了 entrypoint，也可以写成：

```bash
or-cpt-engine <command> [options]
```

本手册统一使用 `python -m or_cpt_engine.cli.main`。

默认生产产物建议放在：

```text
engine_runs/<run_id>/
```

虽然少数 CLI 参数仍有 `data/...` 历史默认值，但当前正式生产建议显式指定 `engine_runs/...`，便于和 Seed Factory / CPT Factory 的 run 目录保持一致。

## 1. 当前命令总览

查看全部命令：

```bash
python -m or_cpt_engine.cli.main --help
```

查看某个命令：

```bash
python -m or_cpt_engine.cli.main run-cpt-from-seeds --help
```

当前命令按用途可以分成 6 组。

| 分组 | 命令 | 用途 |
|---|---|---|
| 环境与资产 | `bootstrap` | 检查配置与环境 |
| 环境与资产 | `import-optmath-generators` | 从 upstream OptMATH 导入 generator 到内部目录 |
| 环境与资产 | `scan-generators` | 扫描内部 generator，生成 registry |
| 环境与资产 | `profile-generators` | 生成或刷新 generator profile |
| 环境与资产 | `smoke-test-generators` | 测试 generator adapter 是否可用 |
| 环境与资产 | `family-contract-inventory` | 输出 family contract 清单 |
| Seed 生产 | `generate-instances` | 阶段 03，生成 seed instance |
| Seed 生产 | `validate-solver` | 阶段 04，Gurobi 验证 |
| Seed 生产 | `validate-instance-quality` | 阶段 04b，seed 质量门 |
| Seed 生产 | `audit-generators` | 一次性跑 03 -> 04 -> 04b，并生成质量画像 |
| 生产计划 | `analyze-generator-quality` | 生成 generator 质量画像 |
| 生产计划 | `plan-production` | 生成 production plan |
| 主工作流 | `run-seed-factory` | 正式 seed 工厂，跑 00 -> 04b |
| 主工作流 | `run-cpt-from-seeds` | 正式 CPT 工厂，从冻结 seed 跑 05 -> 10 |
| 主工作流 | `run-full` | 小规模端到端 smoke / 实验 |
| CPT 阶段 | `backtranslate` | 阶段 05，seed 到业务题面 |
| CPT 阶段 | `filter-nl` | 阶段 06，题面过滤 |
| CPT 阶段 | `forward-model` | 阶段 07，题面到 formulation/code |
| CPT 阶段 | `forward-eval` | 阶段 08，执行 LLM code 并对齐目标值 |
| CPT 阶段 | `render-cpt` | 阶段 09，渲染 CPT Markdown |
| CPT 阶段 | `export` | 阶段 10，train-only export |
| 复盘与优化 | `analyze-run-quality` | 生成 run quality dashboard |
| 复盘与优化 | `analyze-diversity` | 生成 diversity dashboard |
| 复盘与优化 | `optimize-prompts` | 根据失败样本生成 prompt 优化候选 |

## 2. 推荐正式生产顺序

当前最推荐的生产顺序是：

```text
scan-generators
-> smoke-test-generators
-> audit-generators
-> analyze-generator-quality
-> plan-production
-> run-seed-factory
-> run-cpt-from-seeds
-> analyze-run-quality / analyze-diversity
```

原因：

- `audit-generators` 和 `run-seed-factory` 不调用 LLM，可以先把 seed 质量做干净。
- `run-cpt-from-seeds` 只消费 04b 通过的 frozen seed，减少 LLM 浪费。
- Seed run 和 CPT run 各自有 manifest，便于交接和复盘。
- `run-full` 仍然可用，但更适合 smoke 或小实验。

### 2.1 一套可复制的推荐流程

先设置几个 run id：

```bash
REGISTRY_RUN_ID=or_cpt_registry_v1
AUDIT_RUN_ID=seed_audit_all53_x50_v1
SEED_RUN_ID=seed_factory_1k_from_x50_v1
CPT_RUN_ID=cpt_factory_1k_from_seed_v1
```

扫描内部 generator：

```bash
python -m or_cpt_engine.cli.main scan-generators \
  --generators-dir or_cpt_engine/generators/optmath_seed \
  --profiles engine_configs/generator_profiles.yaml \
  --source-label optmath_internal \
  --output engine_runs/${REGISTRY_RUN_ID}/01_generator_registry
```

Smoke test：

```bash
python -m or_cpt_engine.cli.main smoke-test-generators \
  --registry engine_runs/${REGISTRY_RUN_ID}/01_generator_registry/generator_registry.jsonl \
  --output engine_runs/${REGISTRY_RUN_ID}/02_adapter_smoke_test \
  --extract-model-info
```

Seed audit，每个 generator 50 条：

```bash
python -m or_cpt_engine.cli.main audit-generators \
  --registry engine_runs/${REGISTRY_RUN_ID}/01_generator_registry/generator_registry.jsonl \
  --smoke-report engine_runs/${REGISTRY_RUN_ID}/02_adapter_smoke_test/adapter_smoke_report.jsonl \
  --output engine_runs/${AUDIT_RUN_ID} \
  --samples-per-generator 50 \
  --run-id ${AUDIT_RUN_ID} \
  --solver-concurrency 24 \
  --solver-threads-per-process 1
```

生成 generator 质量画像：

```bash
python -m or_cpt_engine.cli.main analyze-generator-quality \
  --run-dir engine_runs/${AUDIT_RUN_ID} \
  --output engine_runs/${AUDIT_RUN_ID}/reports/generator_quality_seed \
  --profiles engine_configs/generator_profiles.yaml
```

生成 10k production plan：

```bash
python -m or_cpt_engine.cli.main plan-production \
  --target-accepted 10000 \
  --profiles engine_configs/generator_profiles.yaml \
  --quality-report engine_runs/${AUDIT_RUN_ID}/reports/generator_quality_seed/generator_quality_report.jsonl \
  --output engine_runs/${AUDIT_RUN_ID}/reports/production_plan_10k
```

按 plan 比例先生产 1000 条 seed attempts：

```bash
python -m or_cpt_engine.cli.main run-seed-factory \
  --run-id ${SEED_RUN_ID} \
  --production-plan engine_runs/${AUDIT_RUN_ID}/reports/production_plan_10k/production_plan.json \
  --num-instances 1000 \
  --solver-concurrency 24 \
  --solver-threads-per-process 1 \
  --log-level INFO
```

从 04b 通过的 seed 生产 CPT：

```bash
python -m or_cpt_engine.cli.main run-cpt-from-seeds \
  --run-id ${CPT_RUN_ID} \
  --seed-input engine_runs/${SEED_RUN_ID}/04b_instance_quality/quality_validated_instances.jsonl \
  --seed-manifest engine_runs/${SEED_RUN_ID}/reports/seed_manifest.json \
  --llm-concurrency 32 \
  --forward-eval-concurrency 24 \
  --solver-threads-per-process 1 \
  --pipeline-mode streaming \
  --log-level INFO
```

最终训练数据：

```text
engine_runs/${CPT_RUN_ID}/10_train_val_export/train.jsonl
```

## 3. 输出文件速查

### 3.1 Registry / smoke 输出

```text
engine_runs/<registry_run_id>/01_generator_registry/
  generator_registry.jsonl
  invalid_generators.jsonl
  generator_inventory.md

engine_runs/<registry_run_id>/02_adapter_smoke_test/
  adapter_smoke_report.jsonl
  adapter_smoke_summary.md
```

### 3.2 Seed audit / seed factory 输出

```text
engine_runs/<seed_run_id>/03_instance_generation/
  instances_raw.jsonl
  instances_generated.jsonl
  instances_generation_rejected.jsonl
  instance_generation_report.md

engine_runs/<seed_run_id>/04_solver_validation/
  solver_validated_instances.jsonl
  solver_rejected_instances.jsonl
  solver_validation_report.md

engine_runs/<seed_run_id>/04b_instance_quality/
  quality_validated_instances.jsonl
  quality_review_instances.jsonl
  quality_rejected_instances.jsonl
  instance_quality_dashboard.csv
  instance_quality_report.md

engine_runs/<seed_run_id>/reports/
  seed_manifest.json
  seed_manifest.md
  generator_quality_report.jsonl
  generator_quality_summary.md
  production_plan_used/
```

最重要的 seed 交接文件：

```text
engine_runs/<seed_run_id>/04b_instance_quality/quality_validated_instances.jsonl
engine_runs/<seed_run_id>/reports/seed_manifest.json
```

### 3.3 CPT factory 输出

```text
engine_runs/<cpt_run_id>/05_backtranslation/
  backtranslation_candidates.jsonl
  backtranslation_rejected.jsonl
  backtranslation_raw_responses.jsonl
  backtranslation_report.md

engine_runs/<cpt_run_id>/06_nl_quality_filter/
  nl_validated_candidates.jsonl
  nl_rejected.jsonl
  nl_quality_report.md

engine_runs/<cpt_run_id>/07_forward_modeling/
  forward_modeling_outputs.jsonl
  forward_modeling_rejected.jsonl
  forward_modeling_raw_responses.jsonl
  forward_repair_outputs.jsonl
  forward_repair_rejected.jsonl
  forward_repair_raw_responses.jsonl
  extracted_code/

engine_runs/<cpt_run_id>/08_forward_eval/
  accepted_pairs.jsonl
  rejected_pairs.jsonl
  forward_eval_report.md
  forward_eval_rejection_dashboard.md
  forward_repair_attempts.jsonl
  forward_repair_report.md

engine_runs/<cpt_run_id>/09_cpt_rendering/
  cpt_documents.jsonl
  cpt_rendering_rejected.jsonl
  cpt_rendering_report.md

engine_runs/<cpt_run_id>/10_train_val_export/
  train.jsonl
  manifest.json
  train_manifest.json
  export_report.md

engine_runs/<cpt_run_id>/reports/
  cpt_run_manifest.json
  cpt_run_manifest.md
  run_quality_dashboard.md
  diversity/diversity_dashboard.md
```

注意：

```text
10_train_val_export 是历史兼容目录名；当前只导出 train.jsonl，不导出 val.jsonl / test.jsonl。
```

## 4. 环境与资产命令

### 4.1 `bootstrap`

用途：检查项目配置、路径、Gurobi / LLM 等运行环境。

常用命令：

```bash
python -m or_cpt_engine.cli.main bootstrap \
  --output engine_runs/bootstrap_check_v1/00_bootstrap
```

核心输出：

```text
environment_report.json
environment_report.md
```

常用选项：

| 选项 | 说明 |
|---|---|
| `--project` | 默认 `engine_configs/project.yaml` |
| `--paths` | 默认 `engine_configs/paths.yaml` |
| `--llm` | 默认 `engine_configs/llm.yaml` |
| `--generation-plan` | 默认 `engine_configs/generation_plan.yaml` |
| `--quality-thresholds` | 默认 `engine_configs/quality_thresholds.yaml` |
| `--rendering` | 默认 `engine_configs/cpt_rendering.yaml` |
| `--split` | 默认 `engine_configs/train_val_split.yaml`，当前主要是历史兼容 |
| `--output` | bootstrap 报告输出目录 |

### 4.2 `scan-generators`

用途：扫描内部 generator 目录，生成 registry。

推荐命令：

```bash
python -m or_cpt_engine.cli.main scan-generators \
  --generators-dir or_cpt_engine/generators/optmath_seed \
  --profiles engine_configs/generator_profiles.yaml \
  --source-label optmath_internal \
  --output engine_runs/or_cpt_registry_v1/01_generator_registry
```

核心输出：

```text
generator_registry.jsonl
invalid_generators.jsonl
generator_inventory.md
```

常用选项：

| 选项 | 说明 |
|---|---|
| `--generators-dir` | 默认 `or_cpt_engine/generators/optmath_seed` |
| `--output` | registry 输出目录；建议显式放到 `engine_runs/...` |
| `--source-label` | registry 来源标签，例如 `optmath_internal` |
| `--profiles` | profile 路径，用于标记 profile 覆盖状态 |

### 4.3 `smoke-test-generators`

用途：动态加载 generator，测试 adapter 是否能生成实例。

推荐命令：

```bash
python -m or_cpt_engine.cli.main smoke-test-generators \
  --registry engine_runs/or_cpt_registry_v1/01_generator_registry/generator_registry.jsonl \
  --output engine_runs/or_cpt_registry_v1/02_adapter_smoke_test \
  --extract-model-info
```

只测一组 generator：

```bash
python -m or_cpt_engine.cli.main smoke-test-generators \
  --registry engine_runs/or_cpt_registry_v1/01_generator_registry/generator_registry.jsonl \
  --output engine_runs/or_cpt_registry_v1/02_adapter_smoke_test \
  --generator-ids optmath_cflp,optmath_transp,optmath_setcover
```

常用选项：

| 选项 | 说明 |
|---|---|
| `--registry` | 必填，`generator_registry.jsonl` |
| `--output` | 必填，smoke report 输出目录 |
| `--seed` | smoke test 随机种子，默认 42 |
| `--difficulty-level` | 传给 adapter 的难度标签，默认 `level_1` |
| `--limit` | 最多测试多少个 generator |
| `--extract-model-info` | 同时抽取/求解模型信息，较慢但更充分 |
| `--generator-ids` | 逗号分隔的 generator id |

### 4.4 `import-optmath-generators`

用途：把 upstream `external/OptMATH/generators` 导入到项目内部目录。

先 dry run：

```bash
python -m or_cpt_engine.cli.main import-optmath-generators \
  --source external/OptMATH/generators \
  --output or_cpt_engine/generators/optmath_seed \
  --dry-run
```

确认后再真实导入：

```bash
python -m or_cpt_engine.cli.main import-optmath-generators \
  --source external/OptMATH/generators \
  --output or_cpt_engine/generators/optmath_seed
```

常用选项：

| 选项 | 说明 |
|---|---|
| `--source` | upstream OptMATH generator 目录，默认 `external/OptMATH/generators` |
| `--output` | 内部 generator 目录，默认 `or_cpt_engine/generators/optmath_seed` |
| `--dry-run` | 只预览，不复制文件 |
| `--overwrite` | 覆盖已有内部 generator 文件，谨慎使用 |

安全提醒：

- 日常优化 generator 源码时，优先改 `or_cpt_engine/generators/optmath_seed/`。
- 不要随手覆盖内部 generator，除非确认 upstream 版本就是你想要的。

### 4.5 `profile-generators`

用途：根据 registry 生成或刷新 `generator_profiles.yaml`。

命令：

```bash
python -m or_cpt_engine.cli.main profile-generators \
  --registry engine_runs/or_cpt_registry_v1/01_generator_registry/generator_registry.jsonl \
  --profiles-output engine_configs/generator_profiles.yaml \
  --report-output engine_runs/or_cpt_registry_v1/reports/generator_profile_inventory.md
```

如需覆盖现有 profile：

```bash
python -m or_cpt_engine.cli.main profile-generators \
  --registry engine_runs/or_cpt_registry_v1/01_generator_registry/generator_registry.jsonl \
  --profiles-output engine_configs/generator_profiles.yaml \
  --report-output engine_runs/or_cpt_registry_v1/reports/generator_profile_inventory.md \
  --overwrite
```

安全提醒：

- 当前 `generator_profiles.yaml` 是生产关键配置。
- 覆盖前建议先备份或确认 diff。
- 正常生产不需要频繁运行这个命令。

### 4.6 `family-contract-inventory`

用途：输出当前 family contract 清单，便于检查 task family、generator override 和契约覆盖。

命令：

```bash
python -m or_cpt_engine.cli.main family-contract-inventory \
  --output engine_runs/reports/family_contract_inventory.md
```

核心输出：

```text
engine_runs/reports/family_contract_inventory.md
```

## 5. Seed 生产命令

### 5.1 `audit-generators`

用途：快速审计 generator 健康度，一次性跑：

```text
03_instance_generation
-> 04_solver_validation
-> 04b_instance_quality
-> analyze-generator-quality
```

它不调用 LLM。

推荐命令：

```bash
python -m or_cpt_engine.cli.main audit-generators \
  --registry engine_runs/or_cpt_registry_v1/01_generator_registry/generator_registry.jsonl \
  --smoke-report engine_runs/or_cpt_registry_v1/02_adapter_smoke_test/adapter_smoke_report.jsonl \
  --output engine_runs/seed_audit_all53_x50_v1 \
  --samples-per-generator 50 \
  --run-id seed_audit_all53_x50_v1 \
  --solver-concurrency 24 \
  --solver-threads-per-process 1
```

只审计部分 generator：

```bash
python -m or_cpt_engine.cli.main audit-generators \
  --registry engine_runs/or_cpt_registry_v1/01_generator_registry/generator_registry.jsonl \
  --smoke-report engine_runs/or_cpt_registry_v1/02_adapter_smoke_test/adapter_smoke_report.jsonl \
  --output engine_runs/seed_audit_core_x100_v1 \
  --samples-per-generator 100 \
  --run-id seed_audit_core_x100_v1 \
  --generator-ids optmath_cflp,optmath_transp,optmath_setcover,optmath_vrptw \
  --solver-concurrency 24 \
  --solver-threads-per-process 1
```

常用选项：

| 选项 | 说明 |
|---|---|
| `--registry` | 必填，registry 文件 |
| `--smoke-report` | 必填，smoke report 文件 |
| `--output` | 必填，audit run 目录 |
| `--samples-per-generator` | 每个 generator 生成多少条，默认 200 |
| `--run-id` | 写入实例 metadata 的 run id，默认 `generator_audit` |
| `--generator-ids` | 只审计指定 generator |
| `--difficulty-mix` | 覆盖难度比例，如 `level_1=0.6,level_2=0.3,level_3=0.1` |
| `--solver-concurrency` | 04 阶段 solver 子进程并发 |
| `--solver-threads-per-process` | 每个 Gurobi 子进程线程数 |

重点查看：

```text
engine_runs/<audit_run_id>/04b_instance_quality/quality_validated_instances.jsonl
engine_runs/<audit_run_id>/04b_instance_quality/quality_review_instances.jsonl
engine_runs/<audit_run_id>/04b_instance_quality/quality_rejected_instances.jsonl
engine_runs/<audit_run_id>/reports/generator_quality_report.jsonl
engine_runs/<audit_run_id>/reports/generator_quality_summary.md
```

### 5.2 `run-seed-factory`

用途：正式 seed 工厂，跑：

```text
00_bootstrap
-> 01_generator_registry
-> 02_adapter_smoke_test
-> 03_instance_generation
-> 04_solver_validation
-> 04b_instance_quality
-> generator quality analysis
-> seed manifest
```

它不调用 LLM。

按 production plan 缩放到 1000 条 seed attempts：

```bash
python -m or_cpt_engine.cli.main run-seed-factory \
  --run-id seed_factory_1k_from_x50_v1 \
  --production-plan engine_runs/seed_audit_all53_x50_v1/reports/production_plan_10k/production_plan.json \
  --num-instances 1000 \
  --solver-concurrency 24 \
  --solver-threads-per-process 1 \
  --log-level INFO
```

不用 production plan，只跑指定 generator：

```bash
python -m or_cpt_engine.cli.main run-seed-factory \
  --run-id seed_factory_core_200_v1 \
  --num-instances 200 \
  --generator-ids optmath_cflp,optmath_transp,optmath_setcover,optmath_vrptw \
  --solver-concurrency 24 \
  --solver-threads-per-process 1 \
  --log-level INFO
```

常用选项：

| 选项 | 说明 |
|---|---|
| `--run-id` | seed run id；不传会自动生成 |
| `--num-instances` | seed attempts 总数；和 production plan 同时使用时按比例缩放 |
| `--production-plan` | production_plan.json |
| `--generator-ids` | 限定 generator 集合 |
| `--solver-concurrency` | 04 阶段 solver 子进程并发 |
| `--solver-threads-per-process` | 每个 Gurobi 子进程线程数 |
| `--log-level` | `DEBUG`、`INFO`、`WARNING`、`ERROR` |
| `--force` | 删除已有同名 run 目录后重新开始，谨慎使用 |

核心输出：

```text
engine_runs/<seed_run_id>/04b_instance_quality/quality_validated_instances.jsonl
engine_runs/<seed_run_id>/reports/seed_manifest.json
engine_runs/<seed_run_id>/reports/production_plan_used/production_plan_usage.json
```

### 5.3 `generate-instances`

用途：只跑阶段 03，手动生成 seed instances。通常用于 debug，正式生产更推荐 `audit-generators` 或 `run-seed-factory`。

命令：

```bash
python -m or_cpt_engine.cli.main generate-instances \
  --registry engine_runs/or_cpt_registry_v1/01_generator_registry/generator_registry.jsonl \
  --smoke-report engine_runs/or_cpt_registry_v1/02_adapter_smoke_test/adapter_smoke_report.jsonl \
  --output engine_runs/manual_stage_run/03_instance_generation \
  --run-id manual_stage_run \
  --num-instances-per-generator 2 \
  --generator-ids optmath_cflp,optmath_transp \
  --profiles engine_configs/generator_profiles.yaml \
  --difficulty-mix level_1=0.6,level_2=0.3,level_3=0.1
```

常用选项：

| 选项 | 说明 |
|---|---|
| `--registry` | 必填，registry 文件 |
| `--smoke-report` | 必填，smoke report 文件 |
| `--output` | 必填，03 输出目录 |
| `--run-id` | 写入实例 metadata 的 run id，默认 `manual` |
| `--num-instances-per-generator` | 每个 smoke-passing generator 生成数量，默认 1 |
| `--limit` | 最多生成 attempts 总数 |
| `--generator-ids` | 限定 generator 集合 |
| `--use-profiles / --no-use-profiles` | 是否使用 profile 注入 difficulty/concept metadata |
| `--profiles` | profile 文件，默认 `engine_configs/generator_profiles.yaml` |
| `--difficulty-mix` | 覆盖难度比例 |
| `--extract-solve-timeout-sec` | 提取模型信息时的 Gurobi 超时，默认 60 |

输出：

```text
instances_raw.jsonl
instances_generated.jsonl
instances_generation_rejected.jsonl
instance_generation_report.md
```

### 5.4 `validate-solver`

用途：只跑阶段 04，用 Gurobi 验证 `instances_generated.jsonl`。

命令：

```bash
python -m or_cpt_engine.cli.main validate-solver \
  --input engine_runs/manual_stage_run/03_instance_generation/instances_generated.jsonl \
  --output engine_runs/manual_stage_run/04_solver_validation \
  --timeout-seconds 60 \
  --concurrency 24 \
  --threads-per-process 1
```

输出：

```text
solver_validated_instances.jsonl
solver_rejected_instances.jsonl
solver_validation_report.md
```

### 5.5 `validate-instance-quality`

用途：只跑阶段 04b，对 solver-validated seed 做质量门。

命令：

```bash
python -m or_cpt_engine.cli.main validate-instance-quality \
  --input engine_runs/manual_stage_run/04_solver_validation/solver_validated_instances.jsonl \
  --output engine_runs/manual_stage_run/04b_instance_quality
```

输出：

```text
quality_validated_instances.jsonl
quality_review_instances.jsonl
quality_rejected_instances.jsonl
instance_quality_dashboard.csv
instance_quality_report.md
```

## 6. 生产计划命令

### 6.1 `analyze-generator-quality`

用途：从一个 seed run、audit run 或 full run 中生成 generator 质量画像。

命令：

```bash
python -m or_cpt_engine.cli.main analyze-generator-quality \
  --run-dir engine_runs/seed_audit_all53_x50_v1 \
  --output engine_runs/seed_audit_all53_x50_v1/reports/generator_quality_seed \
  --profiles engine_configs/generator_profiles.yaml
```

输出：

```text
generator_quality_report.jsonl
generator_quality_dashboard.csv
generator_quality_summary.md
```

如果不传 `--output`，默认输出到：

```text
<run_dir>/reports/generator_quality/
```

### 6.2 `plan-production`

用途：根据 profile 和可选质量报告生成生产配额。

命令：

```bash
python -m or_cpt_engine.cli.main plan-production \
  --target-accepted 10000 \
  --profiles engine_configs/generator_profiles.yaml \
  --quality-report engine_runs/seed_audit_all53_x50_v1/reports/generator_quality_seed/generator_quality_report.jsonl \
  --output engine_runs/seed_audit_all53_x50_v1/reports/production_plan_10k
```

没有质量报告时只按 profile 规划：

```bash
python -m or_cpt_engine.cli.main plan-production \
  --target-accepted 10000 \
  --profiles engine_configs/generator_profiles.yaml \
  --output engine_runs/manual_production_plan_10k
```

常用选项：

| 选项 | 说明 |
|---|---|
| `--target-accepted` | 必填，目标 accepted CPT 文档数 |
| `--output` | 输出目录，含 `production_plan.json/md` |
| `--profiles` | profile 文件 |
| `--quality-report` | 可选，`generator_quality_report.jsonl` |
| `--min-expected-acceptance-rate` | 估算 attempts 的最低通过率，默认 0.05 |

输出：

```text
production_plan.json
production_plan.md
```

注意：

- production plan 是配额表，不是数据集。
- `run-seed-factory --production-plan` 会消费 `production_plan.json`。
- 同时传 `--num-instances` 时会按 plan 比例缩放。

## 7. CPT 生产命令

### 7.1 `run-cpt-from-seeds`

用途：正式 CPT 工厂，从冻结 seed 开始跑 05 -> 10。默认 `--pipeline-mode streaming`。

推荐命令：

```bash
python -m or_cpt_engine.cli.main run-cpt-from-seeds \
  --run-id cpt_factory_1k_from_seed_v1 \
  --seed-input engine_runs/seed_factory_1k_from_x50_v1/04b_instance_quality/quality_validated_instances.jsonl \
  --seed-manifest engine_runs/seed_factory_1k_from_x50_v1/reports/seed_manifest.json \
  --llm-concurrency 32 \
  --forward-eval-concurrency 24 \
  --solver-threads-per-process 1 \
  --pipeline-mode streaming \
  --log-level INFO
```

只消费前 50 条 seed 做 smoke：

```bash
python -m or_cpt_engine.cli.main run-cpt-from-seeds \
  --run-id cpt_smoke_50_v1 \
  --seed-input engine_runs/seed_factory_1k_from_x50_v1/04b_instance_quality/quality_validated_instances.jsonl \
  --seed-manifest engine_runs/seed_factory_1k_from_x50_v1/reports/seed_manifest.json \
  --limit 50 \
  --llm-concurrency 8 \
  --forward-eval-concurrency 8 \
  --pipeline-mode streaming
```

不调用真实 LLM 的 mock 流程：

```bash
python -m or_cpt_engine.cli.main run-cpt-from-seeds \
  --run-id cpt_mock_from_seed_v1 \
  --seed-input engine_runs/seed_factory_1k_from_x50_v1/04b_instance_quality/quality_validated_instances.jsonl \
  --limit 10 \
  --mock-llm \
  --pipeline-mode streaming
```

常用选项：

| 选项 | 说明 |
|---|---|
| `--seed-input` | 必填，通常是 `quality_validated_instances.jsonl` |
| `--run-id` | CPT run id；不传会自动生成 |
| `--seed-manifest` | 可选，复制到 CPT run reports 中记录 lineage |
| `--limit` | 最多消费多少条 seed |
| `--mock-llm` | 不调用真实模型 |
| `--llm-concurrency` | 每个 endpoint 的 LLM 并发覆盖 |
| `--pipeline-mode` | `streaming` 或 `staged`，默认 `streaming` |
| `--forward-eval-concurrency` | streaming 模式下 08 阶段 solver 并发 |
| `--solver-threads-per-process` | 08 阶段每个 solver 子进程线程数 |
| `--pipeline-queue-size` | streaming 队列大小 |
| `--auto-optimize-prompts` | run 结束后做 prompt 优化分析 |
| `--auto-promote-prompts` | 允许自动写回 prompt，谨慎使用 |
| `--force` | 删除已有同名 run 目录后重跑，谨慎使用 |

核心输出：

```text
engine_runs/<cpt_run_id>/10_train_val_export/train.jsonl
engine_runs/<cpt_run_id>/reports/cpt_run_manifest.md
engine_runs/<cpt_run_id>/reports/run_quality_dashboard.md
engine_runs/<cpt_run_id>/reports/diversity/diversity_dashboard.md
```

### 7.2 `run-full`

用途：端到端跑 00 -> 10。当前更适合 smoke、小实验或局部验证，不推荐作为正式大生产入口。

真实小规模 smoke：

```bash
python -m or_cpt_engine.cli.main run-full \
  --run-id or_cpt_smoke_real_v1 \
  --num-instances 20 \
  --generator-ids optmath_cflp,optmath_transp,optmath_setcover \
  --solver-concurrency 8 \
  --solver-threads-per-process 1 \
  --llm-concurrency 8
```

mock smoke：

```bash
python -m or_cpt_engine.cli.main run-full \
  --run-id or_cpt_smoke_mock_v1 \
  --num-instances 10 \
  --generator-ids optmath_cflp,optmath_transp \
  --mock-llm
```

按 production plan 跑小实验：

```bash
python -m or_cpt_engine.cli.main run-full \
  --run-id or_cpt_plan_smoke_100_v1 \
  --production-plan engine_runs/seed_audit_all53_x50_v1/reports/production_plan_10k/production_plan.json \
  --num-instances 100 \
  --solver-concurrency 12 \
  --solver-threads-per-process 1 \
  --llm-concurrency 12
```

常用选项：

| 选项 | 说明 |
|---|---|
| `--run-id` | run id |
| `--num-instances` | 总 seed attempts；不传时按配置默认每 generator 数量 |
| `--production-plan` | 可选 production plan；可与 `--num-instances` 一起缩放 |
| `--mock-llm` | 不调用真实模型 |
| `--generator-ids` | 限定 generator 集合 |
| `--solver-concurrency` | 04 阶段 solver 并发 |
| `--solver-threads-per-process` | 每个 solver 子进程 Gurobi 线程数 |
| `--llm-concurrency` | 每 endpoint LLM 并发 |
| `--auto-optimize-prompts` | 结束后做 prompt 优化分析 |
| `--auto-promote-prompts` | 允许自动写回 prompt，谨慎使用 |

## 8. CPT 拆阶段命令

拆阶段适合调试。正式生产优先用 `run-cpt-from-seeds`。

### 8.1 `backtranslate`

用途：阶段 05，seed 到业务自然语言题面。

正式推荐输入 04b 输出：

```bash
python -m or_cpt_engine.cli.main backtranslate \
  --input engine_runs/manual_stage_run/04b_instance_quality/quality_validated_instances.jsonl \
  --output engine_runs/manual_stage_run/05_backtranslation \
  --llm-concurrency 16
```

也可以使用 mock：

```bash
python -m or_cpt_engine.cli.main backtranslate \
  --input engine_runs/manual_stage_run/04b_instance_quality/quality_validated_instances.jsonl \
  --output engine_runs/manual_stage_run/05_backtranslation \
  --limit 20 \
  --mock-llm
```

输出：

```text
backtranslation_candidates.jsonl
backtranslation_rejected.jsonl
backtranslation_raw_responses.jsonl
backtranslation_report.md
```

### 8.2 `filter-nl`

用途：阶段 06，过滤自然语言题面。

命令：

```bash
python -m or_cpt_engine.cli.main filter-nl \
  --input engine_runs/manual_stage_run/05_backtranslation/backtranslation_candidates.jsonl \
  --output engine_runs/manual_stage_run/06_nl_quality_filter \
  --min-chars 160 \
  --max-chars 4000 \
  --min-number-coverage 0.4
```

输出：

```text
nl_validated_candidates.jsonl
nl_rejected.jsonl
nl_quality_report.md
```

### 8.3 `forward-model`

用途：阶段 07，从 NL 题面生成建模解释、数学模型和 Gurobi code。

命令：

```bash
python -m or_cpt_engine.cli.main forward-model \
  --input engine_runs/manual_stage_run/06_nl_quality_filter/nl_validated_candidates.jsonl \
  --output engine_runs/manual_stage_run/07_forward_modeling \
  --llm-concurrency 16
```

输出：

```text
forward_modeling_outputs.jsonl
forward_modeling_rejected.jsonl
forward_modeling_raw_responses.jsonl
extracted_code/
```

### 8.4 `forward-eval`

用途：阶段 08，执行 LLM 生成的 Gurobi code，并与 reference objective 对齐。

命令：

```bash
python -m or_cpt_engine.cli.main forward-eval \
  --input engine_runs/manual_stage_run/07_forward_modeling/forward_modeling_outputs.jsonl \
  --output engine_runs/manual_stage_run/08_forward_eval \
  --timeout-seconds 60 \
  --abs-tolerance 0.0001 \
  --rel-tolerance 0.0001
```

输出：

```text
accepted_pairs.jsonl
rejected_pairs.jsonl
forward_eval_report.md
forward_eval_rejection_dashboard.md
```

### 8.5 `render-cpt`

用途：阶段 09，把 accepted pair 渲染成 CPT Markdown 文档。

命令：

```bash
python -m or_cpt_engine.cli.main render-cpt \
  --input engine_runs/manual_stage_run/08_forward_eval/accepted_pairs.jsonl \
  --output engine_runs/manual_stage_run/09_cpt_rendering
```

输出：

```text
cpt_documents.jsonl
cpt_rendering_rejected.jsonl
cpt_rendering_report.md
```

### 8.6 `export`

用途：阶段 10，train-only export。

命令：

```bash
python -m or_cpt_engine.cli.main export \
  --input engine_runs/manual_stage_run/09_cpt_rendering/cpt_documents.jsonl \
  --output engine_runs/manual_stage_run/10_train_val_export
```

关闭 exact dedup：

```bash
python -m or_cpt_engine.cli.main export \
  --input engine_runs/manual_stage_run/09_cpt_rendering/cpt_documents.jsonl \
  --output engine_runs/manual_stage_run/10_train_val_export \
  --no-exact-dedup
```

输出：

```text
train.jsonl
manifest.json
train_manifest.json
export_report.md
```

## 9. 复盘、看板与 prompt 优化

### 9.1 `analyze-run-quality`

用途：生成 CPT run 的质量漏斗、拒绝原因、训练集分布等看板。

命令：

```bash
python -m or_cpt_engine.cli.main analyze-run-quality \
  --run-dir engine_runs/cpt_factory_1k_from_seed_v1 \
  --output engine_runs/cpt_factory_1k_from_seed_v1/reports
```

核心输出：

```text
run_quality_dashboard.md
run_quality_dashboard.csv
run_quality_summary.json
train_distribution.md
rejection_reason_by_generator.csv
generator_stage_funnel.csv
```

### 9.2 `analyze-diversity`

用途：生成文本重复、开头模板、near-duplicate、generator/scenario HHI 等多样性指标。

命令：

```bash
python -m or_cpt_engine.cli.main analyze-diversity \
  --run-dir engine_runs/cpt_factory_1k_from_seed_v1 \
  --output engine_runs/cpt_factory_1k_from_seed_v1/reports/diversity \
  --near-duplicate-threshold 0.72
```

常用选项：

| 选项 | 说明 |
|---|---|
| `--near-duplicate-threshold` | Jaccard 近重复阈值，默认 0.72 |
| `--max-rows-per-similarity-group` | 每个 generator/scenario/doc_type 分组最多抽样行数，默认 80 |
| `--max-reported-pairs` | 最多写出多少对 near-duplicate 示例，默认 100 |

核心输出：

```text
diversity_dashboard.md
diversity_summary.json
near_duplicate_examples.jsonl
```

### 9.3 `optimize-prompts`

用途：根据 run 中的 rejected 样本生成 prompt 优化候选。

只生成报告，不写回：

```bash
python -m or_cpt_engine.cli.main optimize-prompts \
  --run-dir engine_runs/cpt_factory_1k_from_seed_v1 \
  --llm-concurrency 8
```

使用 mock：

```bash
python -m or_cpt_engine.cli.main optimize-prompts \
  --run-dir engine_runs/cpt_factory_1k_from_seed_v1 \
  --mock-llm
```

强制生成候选：

```bash
python -m or_cpt_engine.cli.main optimize-prompts \
  --run-dir engine_runs/cpt_factory_1k_from_seed_v1 \
  --force \
  --sample-limit 10 \
  --llm-concurrency 8
```

允许自动写回 prompt 并 bump 版本：

```bash
python -m or_cpt_engine.cli.main optimize-prompts \
  --run-dir engine_runs/cpt_factory_1k_from_seed_v1 \
  --auto-promote \
  --llm-concurrency 8
```

安全提醒：

- 正式生产建议先看候选 prompt 和报告，不要默认 `--auto-promote`。
- Prompt 变化会影响后续样本分布；写回后建议换新的 `run_id`。

## 10. 常用参数速查

### 10.1 Run 参数

| 参数 | 常见命令 | 说明 |
|---|---|---|
| `--run-id` | `run-seed-factory`、`run-cpt-from-seeds`、`run-full` | run 目录名 |
| `--force` | `run-seed-factory`、`run-cpt-from-seeds` | 删除已有同名 run 目录后重跑，谨慎使用 |
| `--log-level` | 主工作流 | `DEBUG`、`INFO`、`WARNING`、`ERROR` |
| `--limit` | LLM 阶段和 CPT factory | 限制输入行数，适合 smoke |
| `--mock-llm` | LLM 阶段和主工作流 | 不调用真实模型 |

### 10.2 Generator 参数

| 参数 | 说明 |
|---|---|
| `--generator-ids` | 逗号分隔，不要加空格 |
| `--num-instances` | 主工作流中 seed attempts 总数 |
| `--num-instances-per-generator` | `generate-instances` 中每个 generator 数量 |
| `--samples-per-generator` | `audit-generators` 中每个 generator 数量 |
| `--difficulty-mix` | 如 `level_1=0.6,level_2=0.3,level_3=0.1` |
| `--production-plan` | production_plan.json |

### 10.3 Solver 参数

| 参数 | 常见命令 | 说明 |
|---|---|---|
| `--solver-concurrency` | `audit-generators`、`run-seed-factory`、`run-full` | 04 阶段 solver 子进程并发 |
| `--forward-eval-concurrency` | `run-cpt-from-seeds` | streaming 模式下 08 阶段 solver 并发 |
| `--threads-per-process` | `validate-solver` | 每个 solver 子进程 Gurobi 线程数 |
| `--solver-threads-per-process` | 主工作流 | 每个 solver 子进程 Gurobi 线程数 |
| `--timeout-seconds` | `validate-solver`、`forward-eval` | 单条 solver/code 执行超时 |
| `--abs-tolerance` | `forward-eval` | objective 绝对误差容差 |
| `--rel-tolerance` | `forward-eval` | objective 相对误差容差 |

推荐起点：

```text
solver_concurrency = 24
solver_threads_per_process = 1
```

如果 Gurobi license、CPU、内存和 IO 都稳定，再尝试提高到 32。

### 10.4 LLM 参数

| 参数 | 说明 |
|---|---|
| `--llm-concurrency` | 每个 endpoint 的 LLM 并发覆盖 |
| `--mock-llm` | 不调用真实模型 |
| `--auto-optimize-prompts` | run 结束后生成 prompt 优化分析 |
| `--auto-promote-prompts` | 主工作流中允许 prompt 自动写回 |
| `--auto-promote` | `optimize-prompts` 中允许 prompt 自动写回 |

当前 `engine_configs/llm.yaml` 默认：

```text
endpoint_pool_path = configs/model_api_pool2.yaml
model = dsv4
timeout_sec = 300
max_retries = 3
concurrency_per_endpoint = 32
endpoint_failure_threshold = 2
endpoint_cooldown_seconds = 45
```

## 11. 典型场景配方

### 11.1 只检查环境

```bash
python -m or_cpt_engine.cli.main bootstrap \
  --output engine_runs/bootstrap_check_v1/00_bootstrap
```

### 11.2 只做 generator 健康审计，不花 LLM

```bash
python -m or_cpt_engine.cli.main audit-generators \
  --registry engine_runs/or_cpt_registry_v1/01_generator_registry/generator_registry.jsonl \
  --smoke-report engine_runs/or_cpt_registry_v1/02_adapter_smoke_test/adapter_smoke_report.jsonl \
  --output engine_runs/seed_audit_all53_x100_v1 \
  --samples-per-generator 100 \
  --run-id seed_audit_all53_x100_v1 \
  --solver-concurrency 24 \
  --solver-threads-per-process 1
```

### 11.3 复用同一批 seed 测不同 prompt

```bash
python -m or_cpt_engine.cli.main run-cpt-from-seeds \
  --run-id cpt_prompt_test_v2 \
  --seed-input engine_runs/seed_factory_1k_from_x50_v1/04b_instance_quality/quality_validated_instances.jsonl \
  --seed-manifest engine_runs/seed_factory_1k_from_x50_v1/reports/seed_manifest.json \
  --limit 100 \
  --llm-concurrency 16 \
  --pipeline-mode streaming
```

### 11.4 只重新生成 dashboard

```bash
python -m or_cpt_engine.cli.main analyze-run-quality \
  --run-dir engine_runs/cpt_factory_1k_from_seed_v1 \
  --output engine_runs/cpt_factory_1k_from_seed_v1/reports

python -m or_cpt_engine.cli.main analyze-diversity \
  --run-dir engine_runs/cpt_factory_1k_from_seed_v1 \
  --output engine_runs/cpt_factory_1k_from_seed_v1/reports/diversity
```

## 12. 常见误区

### 12.1 把 production plan 当数据集

错误理解：

```text
production_plan.json 是数据集
```

正确理解：

```text
production_plan.json 是生产配额表，告诉 run-seed-factory 每个 generator 该生产多少 attempts。
```

### 12.2 直接把 `run-full` 当正式大生产入口

`run-full` 可以跑通端到端，但正式生产建议拆分：

```text
run-seed-factory
-> review seed manifest / generator quality
-> run-cpt-from-seeds
```

这样可以复用 seed、减少 LLM 浪费，也更容易定位问题。

### 12.3 忽略 04b

当前 seed 进入 LLM 前推荐先过：

```text
04b_instance_quality/quality_validated_instances.jsonl
```

不要直接把大量 `solver_validated_instances.jsonl` 送入 CPT 工厂，除非你明确知道自己是在调试。

### 12.4 期待 val/test 输出

当前阶段 10 是 train-only export：

```text
10_train_val_export/train.jsonl
```

目录名保留 `train_val` 只是历史兼容；当前不会生成 `val.jsonl` 和 `test.jsonl`。

### 12.5 高并发 solver 又让每个 Gurobi 进程开多线程

建议：

```text
solver_concurrency = 24
solver_threads_per_process = 1
```

先用更多子进程吃 CPU，不要让每个 Gurobi 子进程再抢很多线程。

### 12.6 随手 `--auto-promote`

Prompt 自动写回会改变后续数据分布。正式生产建议先看 optimizer 报告，再决定是否提升 prompt 版本。

## 13. 最终检查清单

Seed run 完成后检查：

```text
04b_instance_quality/quality_validated_instances.jsonl
04b_instance_quality/quality_review_instances.jsonl
04b_instance_quality/quality_rejected_instances.jsonl
reports/seed_manifest.md
reports/generator_quality_summary.md
reports/production_plan_used/production_plan_usage.md
```

CPT run 完成后检查：

```text
10_train_val_export/train.jsonl
10_train_val_export/manifest.json
reports/cpt_run_manifest.md
reports/run_quality_dashboard.md
reports/diversity/diversity_dashboard.md
08_forward_eval/forward_eval_rejection_dashboard.md
```

如果 `train.jsonl` 数量低于预期，优先看：

```text
06_nl_quality_filter/nl_rejected.jsonl
07_forward_modeling/forward_modeling_rejected.jsonl
08_forward_eval/rejected_pairs.jsonl
reports/run_quality_dashboard.md
```

如果重复或模板化风险高，优先看：

```text
reports/diversity/diversity_dashboard.md
reports/diversity/near_duplicate_examples.jsonl
```

最稳的生产习惯：

```text
先 audit seed，再 plan；
先 seed factory，再 CPT factory；
先看 manifest 和 dashboard，再扩大规模。
```
