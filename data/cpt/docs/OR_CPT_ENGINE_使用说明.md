# OR-CPT Engine 使用说明

最后更新：2026-06-08  
适用范围：当前仓库中的 `or_cpt_engine` 代码、`engine_configs/` 配置和 `engine_runs/` 运行产物。  
配套文档：命令细节请看 `OR_CPT_ENGINE_CLI_命令手册.md`；项目摸底和汇报材料请看 `OR_CPT_ENGINE_项目摸底报告.md` 与 `OR_CPT_ENGINE_项目摸底汇报.pptx`。

## 0. 当前口径先说明

`or_cpt_engine` 是面向 OR-CPT 数据生产的工程化数据工厂。它和 `cpt_cleaner` 的分工不同：

| 项目 | 主要职责 |
|---|---|
| `cpt_cleaner` | 将已有 SFT / ChatML 数据清洗成 CPT 风格文本 |
| `or_cpt_engine` | 从 OptMATH seed generator 出发，生产 solver-verified 的 OR-CPT 训练样本 |

一句话理解：

```text
or_cpt_engine 不是“清洗已有答案”，而是“生产可验证的 OR-CPT 数据资产”。
```

当前正式生产推荐使用“双子任务”流程：

```text
scan-generators
-> smoke-test-generators
-> audit-generators
-> analyze-generator-quality
-> plan-production
-> run-seed-factory
-> run-cpt-from-seeds
-> analyze-run-quality / analyze-diversity
-> review / feedback / next plan
```

需要特别注意的当前事实：

- 默认 run 根目录是 `engine_runs`，不是旧文档里常见的 `data/runs`。
- 默认 generator 来源是 `or_cpt_engine/generators/optmath_seed`，不是直接扫 `external/OptMATH/generators`。
- 当前阶段链路包含 `04b_instance_quality`，它是 seed 进入 LLM 前的重要质量门。
- 当前阶段 10 只导出 `train.jsonl`，不再做 `train / val / test` 切分。
- `10_train_val_export` 目录名是历史兼容名，当前语义是 train-only export。
- `train_val_split.yaml` 保留为历史兼容配置，当前 train-only 导出不使用其中比例。
- 当前没有独立 `quality_gate.yaml` 作为配置入口，质量阈值以 `engine_configs/quality_thresholds.yaml` 和代码内的质量门为准。
- `run-full` 仍然可用，但正式生产更推荐拆成 `run-seed-factory` 和 `run-cpt-from-seeds` 两步。

## 1. 最简心智模型

可以把当前 OR-CPT Engine 理解成两段工厂。

### 1.1 Seed Factory：先生产可信 seed

Seed Factory 不调用 LLM，只跑本地代码、OptMATH generator 和 Gurobi：

```text
00_bootstrap
-> 01_generator_registry
-> 02_adapter_smoke_test
-> 03_instance_generation
-> 04_solver_validation
-> 04b_instance_quality
-> generator quality report
-> seed_manifest
```

它回答的问题是：

```text
这些 generator 能不能稳定生产可解、非退化、有训练价值的优化实例？
```

最终交付给第二段工厂的是：

```text
engine_runs/<seed_run_id>/04b_instance_quality/quality_validated_instances.jsonl
engine_runs/<seed_run_id>/reports/seed_manifest.json
```

### 1.2 CPT Factory：再从冻结 seed 生产训练样本

CPT Factory 从通过 04b 的 seed 开始，会调用 LLM 做自然语言反写和正向建模，然后再用 Gurobi 验收：

```text
05_backtranslation
-> 06_nl_quality_filter
-> 07_forward_modeling
-> 08_forward_eval
-> forward_repair（streaming 模式下按配置触发）
-> 09_cpt_rendering
-> 10_train_val_export（当前实际是 train-only）
-> cpt_run_manifest / dashboards
```

它回答的问题是：

```text
LLM 从业务题面重新建模后，生成的 Gurobi 代码能不能求解，并且目标值能不能和 seed reference objective 对齐？
```

最终训练数据在：

```text
engine_runs/<cpt_run_id>/10_train_val_export/train.jsonl
```

### 1.3 为什么要拆成两个工厂

拆分的好处很直接：

- Seed 阶段不消耗 LLM，可以先大批量验证 generator 质量。
- CPT 阶段只消费通过质量门的 seed，减少 LLM 浪费。
- Seed run 和 CPT run 各自有 manifest，便于交接和复盘。
- 如果 CPT 阶段 prompt 或 LLM 出问题，可以复用已冻结 seed 重跑，不必重新生成优化实例。

## 2. 当前关键资产

### 2.1 `generator_profiles.yaml`

路径：

```text
engine_configs/generator_profiles.yaml
```

这是当前 generator 生产档案表。当前仓库事实：

| 项 | 当前值 |
|---|---:|
| 正式 profiles | 53 |
| enabled profiles | 53 |
| high priority | 18 |
| medium priority | 33 |
| cautious priority | 2 |
| task families | 24 |
| difficulty_support | 全部为 `basic` |

它记录每个 generator 的：

- `enabled`：是否参与生产。
- `priority`：生产优先级，当前为 `high`、`medium`、`cautious`。
- `recommended_weight` / `sampling_weight`：生产计划中的基础权重和上限。
- `task_family` / `sub_family`：问题族和子族。
- `modeling_concepts`：该 generator 涉及的建模概念。
- `canonical_math_signature`：该问题的标准数学结构。
- `quality_risks` / `quality_controls` / `quality_controller`：已知风险和质量门。
- `difficulty_config` / `difficulty_support`：难度配置。

重要说明：

- 当前 `difficulty_support` 全部是 `basic`。这表示 difficulty label 会被记录和传递，但大多数 generator 还没有把 `level_1 / level_2 / level_3` 下沉成真实规模参数。
- 如果某个 generator 大规模表现不好，优先考虑降权或设为 `enabled: false`，不要直接删 profile。
- 新增 generator 后应补齐 profile，否则 production plan 无法正确调度。

### 2.2 `scenario_catalog.yaml`

路径：

```text
engine_configs/scenario_catalog.yaml
```

当前事实：

| 项 | 当前值 |
|---|---:|
| catalog version | `v1.2.0` |
| task families | 24 |
| generator entries | 53 |
| scenario variation | enabled |
| min scenario variants per generator | 50 |
| max expanded scenarios per generator | 240 |

它的作用是把数学实例包装成业务题面。场景信息包括：

- 行业 lens。
- 组织 profile。
- planning horizon。
- business trigger。
- narrative angle。
- entity naming style。
- 单位、实体、禁止误设定等提示。

场景选择是确定性的：同一 seed、generator、candidate index 会稳定选择同一类 guidance；不同 seed 和候选又会自然分散到不同业务外壳。

### 2.3 `family_contracts.yaml`

路径：

```text
engine_configs/family_contracts.yaml
```

当前事实：

| 项 | 当前值 |
|---|---:|
| version | `v1.0.0` |
| family contracts | 13 |
| generator overrides | 27 |
| generator aliases | 14 |
| defaults | 1 |

Family contract 的作用是把 OR 建模常识配置化，供 forward modeling prompt 和静态检查使用。例如：

- facility location：开站变量、服务分配、open-before-serve 关系。
- network flow：流守恒、容量、供需平衡。
- routing：访问一次、车辆容量、时间窗。
- lot sizing：库存递推、生产和持有成本。
- covering：覆盖约束和选择变量。
- portfolio：预算、风险、收益。

### 2.4 Prompt 版本

Prompt 版本文件在：

```text
or_cpt_engine/prompts/prompt_versions.yaml
```

当前主要版本：

| prompt | 当前版本 | 作用 |
|---|---|---|
| `backtranslation_prompt` | `v1.0.8` | 常规 seed 到业务题面 |
| `backtranslation_compact_prompt` | `v1.0.1` | 复杂 generator 的紧凑题面生成 |
| `forward_modeling_prompt` | `v1.0.6` | 业务题面到 formulation/code |
| `forward_model_repair_prompt` | `v1.0.0` | 08 失败后的单轮修复 |

LLM 产物会记录 prompt name、version、hash，便于追踪某批数据由哪份 prompt 生成。

## 3. 推荐生产流程

### 3.1 第一步：扫描 generator

推荐显式输出到 `engine_runs`：

```bash
RUN_ID=or_cpt_registry_v1

python -m or_cpt_engine.cli.main scan-generators \
  --generators-dir or_cpt_engine/generators/optmath_seed \
  --profiles engine_configs/generator_profiles.yaml \
  --source-label optmath_internal \
  --output engine_runs/${RUN_ID}/01_generator_registry
```

核心输出：

```text
engine_runs/${RUN_ID}/01_generator_registry/generator_registry.jsonl
engine_runs/${RUN_ID}/01_generator_registry/invalid_generators.jsonl
engine_runs/${RUN_ID}/01_generator_registry/generator_inventory.md
```

### 3.2 第二步：adapter smoke test

```bash
python -m or_cpt_engine.cli.main smoke-test-generators \
  --registry engine_runs/${RUN_ID}/01_generator_registry/generator_registry.jsonl \
  --output engine_runs/${RUN_ID}/02_adapter_smoke_test \
  --extract-model-info
```

核心输出：

```text
engine_runs/${RUN_ID}/02_adapter_smoke_test/adapter_smoke_report.jsonl
engine_runs/${RUN_ID}/02_adapter_smoke_test/adapter_smoke_summary.md
```

如果只想测少数 generator：

```bash
python -m or_cpt_engine.cli.main smoke-test-generators \
  --registry engine_runs/${RUN_ID}/01_generator_registry/generator_registry.jsonl \
  --output engine_runs/${RUN_ID}/02_adapter_smoke_test \
  --generator-ids optmath_cflp,optmath_transp,optmath_setcover
```

### 3.3 第三步：seed audit

Seed audit 只跑 `03 -> 04 -> 04b`，不调用 LLM。

建议正式生产前不要只跑每个 generator 5 条。5 条适合工程 smoke；正式扩大前建议每个 generator 50-100 条。

```bash
AUDIT_RUN_ID=seed_audit_all53_x50_v1

python -m or_cpt_engine.cli.main audit-generators \
  --registry engine_runs/${RUN_ID}/01_generator_registry/generator_registry.jsonl \
  --smoke-report engine_runs/${RUN_ID}/02_adapter_smoke_test/adapter_smoke_report.jsonl \
  --output engine_runs/${AUDIT_RUN_ID} \
  --samples-per-generator 50 \
  --run-id ${AUDIT_RUN_ID} \
  --solver-concurrency 24 \
  --solver-threads-per-process 1
```

重点查看：

```text
engine_runs/${AUDIT_RUN_ID}/03_instance_generation/
engine_runs/${AUDIT_RUN_ID}/04_solver_validation/
engine_runs/${AUDIT_RUN_ID}/04b_instance_quality/
```

04b 通过的 seed 在：

```text
engine_runs/${AUDIT_RUN_ID}/04b_instance_quality/quality_validated_instances.jsonl
```

### 3.4 第四步：generator 质量画像

```bash
python -m or_cpt_engine.cli.main analyze-generator-quality \
  --run-dir engine_runs/${AUDIT_RUN_ID} \
  --output engine_runs/${AUDIT_RUN_ID}/reports/generator_quality_seed \
  --profiles engine_configs/generator_profiles.yaml
```

输出：

```text
engine_runs/${AUDIT_RUN_ID}/reports/generator_quality_seed/generator_quality_report.jsonl
engine_runs/${AUDIT_RUN_ID}/reports/generator_quality_seed/generator_quality_dashboard.csv
engine_runs/${AUDIT_RUN_ID}/reports/generator_quality_seed/generator_quality_summary.md
```

这个报告会按 generator 汇总质量表现，后续 production plan 可以读取它来调整配额。

### 3.5 第五步：生成 production plan

```bash
python -m or_cpt_engine.cli.main plan-production \
  --target-accepted 10000 \
  --profiles engine_configs/generator_profiles.yaml \
  --quality-report engine_runs/${AUDIT_RUN_ID}/reports/generator_quality_seed/generator_quality_report.jsonl \
  --output engine_runs/${AUDIT_RUN_ID}/reports/production_plan_10k
```

输出：

```text
engine_runs/${AUDIT_RUN_ID}/reports/production_plan_10k/production_plan.json
engine_runs/${AUDIT_RUN_ID}/reports/production_plan_10k/production_plan.md
```

注意：

- production plan 是配额表，不是数据集。
- 如果后续 `run-seed-factory` 同时传 `--production-plan` 和 `--num-instances`，系统会按 plan 比例缩放到指定 seed attempts 总数。
- 如果只传 `--production-plan`，会使用 plan 中的 `planned_attempts`。

### 3.6 第六步：运行 Seed Factory

示例：按 10k plan 比例先缩放到 1000 条 seed attempts。

```bash
SEED_RUN_ID=seed_factory_1k_from_x50_v1

python -m or_cpt_engine.cli.main run-seed-factory \
  --run-id ${SEED_RUN_ID} \
  --production-plan engine_runs/${AUDIT_RUN_ID}/reports/production_plan_10k/production_plan.json \
  --num-instances 1000 \
  --solver-concurrency 24 \
  --solver-threads-per-process 1 \
  --log-level INFO
```

核心输出：

```text
engine_runs/${SEED_RUN_ID}/04b_instance_quality/quality_validated_instances.jsonl
engine_runs/${SEED_RUN_ID}/04b_instance_quality/quality_review_instances.jsonl
engine_runs/${SEED_RUN_ID}/04b_instance_quality/quality_rejected_instances.jsonl
engine_runs/${SEED_RUN_ID}/reports/seed_manifest.json
engine_runs/${SEED_RUN_ID}/reports/seed_manifest.md
engine_runs/${SEED_RUN_ID}/reports/production_plan_used/production_plan_usage.json
engine_runs/${SEED_RUN_ID}/reports/generator_quality_report.jsonl
```

### 3.7 第七步：从冻结 seed 运行 CPT Factory

默认推荐 `streaming` 模式。

```bash
CPT_RUN_ID=cpt_factory_1k_from_seed_v1

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

核心输出：

```text
engine_runs/${CPT_RUN_ID}/10_train_val_export/train.jsonl
engine_runs/${CPT_RUN_ID}/10_train_val_export/manifest.json
engine_runs/${CPT_RUN_ID}/10_train_val_export/train_manifest.json
engine_runs/${CPT_RUN_ID}/10_train_val_export/export_report.md
engine_runs/${CPT_RUN_ID}/reports/cpt_run_manifest.json
engine_runs/${CPT_RUN_ID}/reports/cpt_run_manifest.md
engine_runs/${CPT_RUN_ID}/reports/run_quality_dashboard.md
engine_runs/${CPT_RUN_ID}/reports/diversity/diversity_dashboard.md
```

如果只是测试流程，可以加 `--mock-llm`，避免真实调用模型：

```bash
python -m or_cpt_engine.cli.main run-cpt-from-seeds \
  --run-id cpt_mock_from_seed_v1 \
  --seed-input engine_runs/${SEED_RUN_ID}/04b_instance_quality/quality_validated_instances.jsonl \
  --limit 10 \
  --mock-llm \
  --pipeline-mode streaming
```

### 3.8 第八步：复盘 dashboard

如果 `run-cpt-from-seeds` 已经自动生成 dashboard，一般直接看：

```text
engine_runs/${CPT_RUN_ID}/reports/run_quality_dashboard.md
engine_runs/${CPT_RUN_ID}/reports/diversity/diversity_dashboard.md
engine_runs/${CPT_RUN_ID}/08_forward_eval/forward_eval_rejection_dashboard.md
```

也可以单独重跑：

```bash
python -m or_cpt_engine.cli.main analyze-run-quality \
  --run-dir engine_runs/${CPT_RUN_ID} \
  --output engine_runs/${CPT_RUN_ID}/reports

python -m or_cpt_engine.cli.main analyze-diversity \
  --run-dir engine_runs/${CPT_RUN_ID} \
  --output engine_runs/${CPT_RUN_ID}/reports/diversity
```

## 4. `run-full` 什么时候用

`run-full` 会在一个 run 目录里从 00 跑到 10：

```bash
python -m or_cpt_engine.cli.main run-full \
  --run-id or_cpt_smoke_v1 \
  --num-instances 20 \
  --generator-ids optmath_cflp,optmath_transp,optmath_setcover \
  --llm-concurrency 8 \
  --solver-concurrency 8 \
  --solver-threads-per-process 1
```

它适合：

- 小规模端到端 smoke。
- 本地确认新 prompt、新 contract、新 generator 改动是否跑通。
- 临时实验。

它不再是正式生产首选方式。正式生产建议拆成：

```text
run-seed-factory
-> review seed manifest / generator quality
-> run-cpt-from-seeds
```

这样可以减少 LLM 浪费，也更容易复盘。

## 5. 阶段说明与当前产物

### 5.1 阶段总览

| 阶段 | 目录 | 是否调用 LLM | 主要职责 |
|---|---|---:|---|
| 00 | `00_bootstrap` | 否 | 检查配置、路径、环境 |
| 01 | `01_generator_registry` | 否 | 扫描内部 generator |
| 02 | `02_adapter_smoke_test` | 否 | 动态加载和最小生成测试 |
| 03 | `03_instance_generation` | 否 | 生成 seed instance |
| 04 | `04_solver_validation` | 否 | Gurobi 独立求解，得到 reference truth |
| 04b | `04b_instance_quality` | 否 | 过滤退化、重复、低训练价值 seed |
| 05 | `05_backtranslation` | 是 | seed 到业务自然语言题面 |
| 06 | `06_nl_quality_filter` | 否 | 题面质量过滤和防泄漏 |
| 07 | `07_forward_modeling` | 是 | NL 到 formulation、解释和 Gurobi code |
| 08 | `08_forward_eval` | 否 | 执行 LLM code，与 reference objective 对齐 |
| 09 | `09_cpt_rendering` | 否 | 渲染 Markdown CPT 文档 |
| 10 | `10_train_val_export` | 否 | exact dedup 并导出 `train.jsonl` |

只调用 LLM 的主阶段是：

- `05_backtranslation`
- `07_forward_modeling`
- `optimize-prompts`，只有显式运行时才调用

### 5.2 00_bootstrap

输入：

- `engine_configs/project.yaml`
- `engine_configs/paths.yaml`
- `engine_configs/llm.yaml`
- `engine_configs/generation_plan.yaml`
- `engine_configs/quality_thresholds.yaml`
- `engine_configs/cpt_rendering.yaml`
- `engine_configs/train_val_split.yaml`

输出：

```text
environment_report.json
environment_report.md
```

职责：

- 检查配置能否加载。
- 检查关键路径、Gurobi、LLM 配置等环境信息。
- 不生产样本。

### 5.3 01_generator_registry

输入：

```text
or_cpt_engine/generators/optmath_seed
engine_configs/generator_profiles.yaml
```

输出：

```text
generator_registry.jsonl
invalid_generators.jsonl
generator_inventory.md
```

职责：

- 扫描内部 OptMATH seed generator。
- 记录 generator id、来源、profile 覆盖状态、文件信息等。
- 把无法识别或结构异常的目录写入 invalid 文件。

### 5.4 02_adapter_smoke_test

输入：

```text
generator_registry.jsonl
```

输出：

```text
adapter_smoke_report.jsonl
adapter_smoke_summary.md
```

职责：

- 动态加载 generator。
- 进行最小样本生成测试。
- 验证 adapter 是否能提取模型信息。

### 5.5 03_instance_generation

输入：

```text
generator_registry.jsonl
adapter_smoke_report.jsonl
engine_configs/generation_plan.yaml
engine_configs/generator_profiles.yaml
```

输出：

```text
instances_raw.jsonl
instances_generated.jsonl
instances_generation_rejected.jsonl
instance_generation_report.md
```

职责：

- 对 smoke pass 的 generator 批量生成 seed instance。
- 写入 generator/profile 元信息。
- 记录 difficulty level、task family、sub family、concept tags、variant 等。
- 初步过滤无法生成或结构不完整的实例。

### 5.6 04_solver_validation

输入：

```text
instances_generated.jsonl
```

输出：

```text
solver_validated_instances.jsonl
solver_rejected_instances.jsonl
solver_validation_report.md
```

职责：

- 用 Gurobi 对每个 seed 做独立求解。
- 只保留 solver status 可接受且有 objective value 的实例。
- 写入 objective、solution、变量诊断、约束诊断等 reference truth。

并发：

- 默认读取 `generation_plan.yaml` 中的 `filters.solver_concurrency`，当前为 24。
- 每个 solver 子进程线程数读取 `filters.solver_threads_per_process`，当前为 1。
- 可通过 `--solver-concurrency` 和 `--solver-threads-per-process` 覆盖。

### 5.7 04b_instance_quality

输入：

```text
solver_validated_instances.jsonl
engine_configs/quality_thresholds.yaml
engine_configs/generator_profiles.yaml
```

输出：

```text
quality_validated_instances.jsonl
quality_review_instances.jsonl
quality_rejected_instances.jsonl
instance_quality_dashboard.csv
instance_quality_report.md
```

职责：

- 在“solver 可解”之上继续筛选“有训练价值”的 seed。
- 检查退化解、全零解、全部变量在边界、binding 约束不足、非零变量过少。
- 检查重复参数、重复 solution signature。
- 应用 generator/profile 家族质量规则。

分类：

- `quality_validated_instances.jsonl`：推荐进入 CPT Factory。
- `quality_review_instances.jsonl`：可以人工检查或用于调参。
- `quality_rejected_instances.jsonl`：建议丢弃或修 generator。

### 5.8 05_backtranslation

输入：

```text
quality_validated_instances.jsonl
or_cpt_engine/prompts/backtranslation_prompt.md
or_cpt_engine/prompts/backtranslation_compact_prompt.md
engine_configs/scenario_catalog.yaml
engine_configs/llm.yaml
```

输出：

```text
backtranslation_candidates.jsonl
backtranslation_rejected.jsonl
backtranslation_raw_responses.jsonl
backtranslation_report.md
```

职责：

- 调用 LLM，把结构化优化实例改写成自然语言业务题。
- 题面不能泄漏 objective value、solver status、变量解、LP 文本或 Gurobi code。
- 通过 scenario guidance 注入行业、组织、业务触发、实体命名、单位等业务语境。
- 默认每个 seed 生成 2 个候选，来自 `llm.yaml` 的 `backtranslation.num_candidates`。

### 5.9 06_nl_quality_filter

输入：

```text
backtranslation_candidates.jsonl
engine_configs/quality_thresholds.yaml
```

输出：

```text
nl_validated_candidates.jsonl
nl_rejected.jsonl
nl_quality_report.md
```

职责：

- 检查题面长度。
- 检查 forbidden phrases。
- 检查是否泄漏答案、solver status、代码或 ChatML 痕迹。
- 检查业务信号和数字覆盖率。

当前阈值：

- 最短字符数：160。
- 最长字符数：4000。
- 数字覆盖率：0.40。

### 5.10 07_forward_modeling

输入：

```text
nl_validated_candidates.jsonl
or_cpt_engine/prompts/forward_modeling_prompt.md
engine_configs/family_contracts.yaml
engine_configs/generator_profiles.yaml
engine_configs/llm.yaml
```

输出：

```text
forward_modeling_outputs.jsonl
forward_modeling_rejected.jsonl
forward_modeling_raw_responses.jsonl
extracted_code/*.py
```

streaming repair 相关输出：

```text
forward_repair_outputs.jsonl
forward_repair_rejected.jsonl
forward_repair_raw_responses.jsonl
```

职责：

- 调用 LLM，从自然语言题生成：
  - `modeling_explanation`
  - `math_model`
  - `gurobipy_code`
- 提取可执行代码到 `extracted_code/`。
- 做静态代码检查和 family contract 检查。
- 代码必须自包含，不能读外部文件，必须创建并优化 Gurobi model。

### 5.11 08_forward_eval

输入：

```text
forward_modeling_outputs.jsonl
extracted_code/*.py
```

输出：

```text
accepted_pairs.jsonl
rejected_pairs.jsonl
forward_eval_report.md
forward_eval_rejection_dashboard.md
```

streaming repair 相关输出：

```text
forward_repair_attempts.jsonl
forward_repair_report.md
```

职责：

- 执行 LLM 生成的 Gurobi code。
- 解析 generated objective、solution、feasibility 等。
- 与 seed reference objective 做 abs / rel tolerance 比较。
- 只有 solver 成功且 objective 匹配的样本才进入 accepted。

当前容差：

```text
objective_abs_tolerance = 0.0001
objective_rel_tolerance = 0.0001
```

Forward repair：

- 当前 `quality_thresholds.yaml` 中 `forward_repair.enabled=true`。
- 默认最多修复 1 次。
- 修复目标包括 objective mismatch、solver status failure、execution failure。
- repair 后仍必须再次经过 08 eval。

### 5.12 09_cpt_rendering

输入：

```text
accepted_pairs.jsonl
engine_configs/cpt_rendering.yaml
engine_configs/quality_thresholds.yaml
```

输出：

```text
cpt_documents.jsonl
cpt_rendering_rejected.jsonl
cpt_rendering_report.md
```

职责：

- 不调用 LLM。
- 把 accepted pair 渲染成 Markdown CPT 文档。
- 文档包含业务题、建模解释、数学模型、Gurobi 代码、求解验证、目标复算等内容。
- 检查 forbidden phrases、必要章节、最小长度和 Python code block。

当前 doc type policy：

| doc type | ratio |
|---|---:|
| `final_model_report` | 0.50 |
| `modeling_rationale` | 0.50 |
| `solver_validation_note` | 0 |

当前 numeric rendering：

- enabled。
- prose/table 中最多保留 2 位小数。
- 不改代码块内部数值。

### 5.13 10_train_val_export

输入：

```text
cpt_documents.jsonl
```

streaming 模式下由 09 阶段逐条传入已通过渲染检查的 CPT 文档。

输出：

```text
train.jsonl
manifest.json
train_manifest.json
export_report.md
```

职责：

- exact text dedup。
- 只输出 `train.jsonl`。
- 记录 doc type、generator、task family、difficulty、scenario、business trigger、style、variant、concept tag 等分布。

再次强调：

```text
10_train_val_export 是历史目录名；当前不再生成 val.jsonl / test.jsonl。
```

## 6. 流式 pipeline

`run-cpt-from-seeds` 默认使用 `--pipeline-mode streaming`。

核心队列：

```text
bt_queue
-> nl_queue
-> fm_queue
-> eval_queue
-> render_queue
-> streaming train exporter
```

主要特点：

- 05 backtranslation 与 07 forward modeling 使用异步 LLM worker。
- 08 forward eval 使用并发 solver 子进程。
- LLM 瞬时错误会 requeue，不直接当成样本质量失败。
- 09 通过一条，10 就可以追加一条到 `train.jsonl`。
- 队列大小默认是 `max(1024, total_llm_capacity * 8)`，可通过 `--pipeline-queue-size` 覆盖。

什么时候用 staged：

```bash
python -m or_cpt_engine.cli.main run-cpt-from-seeds \
  --seed-input engine_runs/${SEED_RUN_ID}/04b_instance_quality/quality_validated_instances.jsonl \
  --run-id cpt_staged_debug_v1 \
  --pipeline-mode staged
```

`staged` 更适合排查某个阶段的中间文件；正式生产推荐 `streaming`。

## 7. 并发与稳定性

### 7.1 LLM 并发

当前 `engine_configs/llm.yaml`：

```yaml
providers:
  default:
    endpoint_pool_path: configs/model_api_pool2.yaml
    model: dsv4
    timeout_sec: 300
    max_retries: 3
    concurrency_per_endpoint: 32
    endpoint_failure_threshold: 2
    endpoint_cooldown_seconds: 45
```

总 LLM 并发约等于：

```text
endpoint 数量 * concurrency_per_endpoint
```

命令行覆盖：

```bash
python -m or_cpt_engine.cli.main run-cpt-from-seeds \
  --seed-input engine_runs/${SEED_RUN_ID}/04b_instance_quality/quality_validated_instances.jsonl \
  --run-id cpt_with_custom_llm_concurrency \
  --llm-concurrency 16
```

### 7.2 LLM requeue

可重试错误包括：

- timeout。
- transport error。
- HTTP 5xx。
- client retry 耗尽后的瞬时失败。

处理逻辑：

```text
LLM transient failure
-> worker requeue item
-> endpoint pool 根据健康状态继续调度
```

业务质量错误不会 requeue，例如：

- LLM 返回 JSON 结构不对。
- NL 题面质量不通过。
- forward code 静态检查失败。

### 7.3 Solver 并发

当前 `engine_configs/generation_plan.yaml`：

```yaml
filters:
  solve_timeout_sec: 60
  solver_concurrency: 24
  solver_threads_per_process: 1
```

建议：

- 多 vCPU 机器先用 `solver_concurrency=24`、`solver_threads_per_process=1`。
- 如果 license、CPU、内存、IO 都稳定，再试 32。
- 不建议在高并发子进程下让每个 Gurobi 进程再开很多线程。

## 8. 配置文件说明

### 8.1 `engine_configs/paths.yaml`

当前关键字段：

| 字段 | 当前值 | 说明 |
|---|---|---|
| `runs_root` | `engine_runs` | run 输出根目录 |
| `optmath_generators_dir` | `or_cpt_engine/generators/optmath_seed` | 默认内部 generator 来源 |
| `scenario_catalog_path` | `engine_configs/scenario_catalog.yaml` | 场景目录 |
| `family_contracts_path` | `engine_configs/family_contracts.yaml` | 家族契约 |
| `generator_profiles_path` | `engine_configs/generator_profiles.yaml` | generator 档案 |

### 8.2 `engine_configs/generation_plan.yaml`

关键字段：

| 字段 | 当前值 | 说明 |
|---|---:|---|
| `generation.num_instances_per_generator` | 5 | 默认每个 generator 生成数量 |
| `filters.solve_timeout_sec` | 60 | Gurobi 超时 |
| `filters.solver_concurrency` | 24 | solver 子进程并发 |
| `filters.solver_threads_per_process` | 1 | 每个 Gurobi 子进程线程数 |
| `difficulty_mix.level_1` | 0.60 | 难度标签混采 |
| `difficulty_mix.level_2` | 0.30 | 难度标签混采 |
| `difficulty_mix.level_3` | 0.10 | 难度标签混采 |

### 8.3 `engine_configs/quality_thresholds.yaml`

关键字段：

| 字段 | 当前值 |
|---|---:|
| `instance_generation.min_solver_validated_rate` | 0.50 |
| `backtranslation.min_chars` | 160 |
| `backtranslation.max_chars` | 4000 |
| `backtranslation.min_number_coverage` | 0.40 |
| `forward_modeling_eval.objective_abs_tolerance` | 0.0001 |
| `forward_modeling_eval.objective_rel_tolerance` | 0.0001 |
| `forward_repair.enabled` | true |
| `forward_repair.max_attempts` | 1 |
| `cpt_rendering.min_chars` | 400 |

### 8.4 `engine_configs/cpt_rendering.yaml`

关键字段：

- `language: en`。
- `doc_type_policy` 当前为 50% `final_model_report`、50% `modeling_rationale`、0% `solver_validation_note`。
- `max_views_per_seed_instance: 12`。
- `numeric_rendering.enabled: true`。
- `numeric_rendering.max_decimal_places: 2`。
- `numeric_rendering.apply_to_code_blocks: false`。
- `style_policy` 会轮换 verbosity、writing style、audience、section order。

### 8.5 `engine_configs/train_val_split.yaml`

当前只是历史兼容配置。阶段 10 当前不再按它切分 train / val / test。

请不要用这个文件判断当前导出比例。当前最终数据只看：

```text
10_train_val_export/train.jsonl
10_train_val_export/manifest.json
```

## 9. 质量保障体系

当前质量体系分为七层。

| 层 | 阶段 | 主要检查 |
|---|---|---|
| 1 | 02 adapter smoke | generator 能否加载和生成 |
| 2 | 04 solver validation | seed 是否可被 Gurobi 求解 |
| 3 | 04b instance quality | 是否退化、重复、低训练价值 |
| 4 | 06 NL filter | 题面是否业务化、是否泄漏答案 |
| 5 | 07 static / contract gate | 代码结构、外部读取、family contract |
| 6 | 08 objective match | generated objective 是否对齐 reference objective |
| 7 | 09/10 render/export | Markdown 质量、代码块、exact dedup |

### 9.1 Rejection taxonomy

系统会尽量把失败归因到可行动的细分原因。常见类型包括：

- `STATIC_CODE_CHECK_FAILED`
- `STATIC_SIGNATURE_MISMATCH`
- `CONTRACT_MISSING_REQUIRED_CONSTRAINT`
- `FORWARD_SOLVER_NOT_OPTIMAL`
- `OBJECTIVE_MISMATCH`
- `objective_mismatch_likely_missing_fixed_cost`
- `objective_mismatch_likely_balance_or_flow_error`
- `solver_infeasible`
- `execution_syntax_error`
- `execution_type_error`

这套分类的价值是：失败不是黑盒。后续可以按 generator、task family、fine reason 逐项修 prompt、contract 或 generator。

### 9.2 当前真实运行参考

已有 run：

```text
engine_runs/cpt_factory_200v1_v4
```

关键结果：

| 指标 | 数值 |
|---|---:|
| seed input count | 199 |
| train count | 143 |
| export mode | `train_only_streaming` |
| seed-to-train rate | 71.86% |
| NL accept rate | 91.86% |
| forward-eval accept rate | 51.81% |
| rendering rejected | 0 |
| duplicates removed | 0 |
| exact duplicate rows | 0 |
| forbidden phrase hits | 0 |
| missing section hits | 0 |
| missing code block rows | 0 |

对应文件：

```text
engine_runs/cpt_factory_200v1_v4/reports/cpt_run_manifest.md
engine_runs/cpt_factory_200v1_v4/reports/run_quality_dashboard.md
engine_runs/cpt_factory_200v1_v4/reports/diversity/diversity_dashboard.md
```

## 10. 运行产物怎么看

### 10.1 Seed run 典型结构

```text
engine_runs/<seed_run_id>/
  00_bootstrap/
  01_generator_registry/
  02_adapter_smoke_test/
  03_instance_generation/
  04_solver_validation/
  04b_instance_quality/
  logs/
  reports/
    generator_quality_report.jsonl
    generator_quality_summary.md
    production_plan_used/
    seed_manifest.json
    seed_manifest.md
    throughput_summary.md
```

最重要的交接文件：

```text
04b_instance_quality/quality_validated_instances.jsonl
reports/seed_manifest.json
```

### 10.2 CPT run 典型结构

```text
engine_runs/<cpt_run_id>/
  05_backtranslation/
  06_nl_quality_filter/
  07_forward_modeling/
  08_forward_eval/
  09_cpt_rendering/
  10_train_val_export/
  logs/
  reports/
    cpt_run_manifest.json
    cpt_run_manifest.md
    run_quality_dashboard.md
    diversity/
      diversity_dashboard.md
```

最重要的训练输出：

```text
10_train_val_export/train.jsonl
```

### 10.3 `train.jsonl` 大致结构

每条样本通常包含：

```json
{
  "id": "or_cpt_xxx",
  "text": "# Optimization Modeling Report\n...",
  "source_id": "inst_xxx",
  "metadata": {
    "source": "or_cpt_engine",
    "generator_id": "optmath_cflp",
    "task_family": "facility_location",
    "doc_type": "final_model_report",
    "difficulty_level": "level_1"
  }
}
```

字段会随 renderer 和 exporter 增加更多 metadata，例如 scenario、business trigger、writing style、variant、concept tags 等。

## 11. Prompt 优化

单独运行：

```bash
python -m or_cpt_engine.cli.main optimize-prompts \
  --run-dir engine_runs/${CPT_RUN_ID} \
  --llm-concurrency 8
```

默认行为：

- 读取 run 中的 rejected 产物。
- 判断主要问题属于 backtranslation、forward modeling 还是 repair。
- 生成候选 prompt 和报告。
- 默认不写回正式 prompt。

输出位置：

```text
engine_runs/${CPT_RUN_ID}/reports/prompt_optimization/
```

如果确认要自动写回 prompt 并提升版本：

```bash
python -m or_cpt_engine.cli.main optimize-prompts \
  --run-dir engine_runs/${CPT_RUN_ID} \
  --auto-promote \
  --llm-concurrency 8
```

生产建议：

- 小规模 smoke 可以尝试 `--auto-promote`。
- 正式生产建议只生成候选报告，不自动写回。
- Prompt 版本变化后建议换新的 `run_id`。

`run-cpt-from-seeds` 和 `run-full` 也支持运行结束后自动做 prompt 优化分析：

```bash
--auto-optimize-prompts
```

如需允许自动写回：

```bash
--auto-promote-prompts
```

## 12. 常用审计命令

### 12.1 Run quality dashboard

```bash
python -m or_cpt_engine.cli.main analyze-run-quality \
  --run-dir engine_runs/${CPT_RUN_ID} \
  --output engine_runs/${CPT_RUN_ID}/reports
```

### 12.2 Diversity dashboard

```bash
python -m or_cpt_engine.cli.main analyze-diversity \
  --run-dir engine_runs/${CPT_RUN_ID} \
  --output engine_runs/${CPT_RUN_ID}/reports/diversity \
  --near-duplicate-threshold 0.72
```

### 12.3 Family contract inventory

```bash
python -m or_cpt_engine.cli.main family-contract-inventory \
  --output engine_runs/reports/family_contract_inventory.md
```

## 13. 常见问题

### Q1：为什么目录叫 `10_train_val_export`，却没有 val/test？

这是历史兼容目录名。当前实现只输出 `train.jsonl`、`manifest.json`、`train_manifest.json` 和 `export_report.md`。不要再期待 `val.jsonl` 或 `test.jsonl`。

### Q2：为什么正式生产不建议直接跑 `run-full`？

因为 `run-full` 会把 seed 生产和 LLM 生产放在同一个 run 里。正式生产时先冻结 seed，再从 seed 生产 CPT，更省 LLM，也更利于复盘和重跑。

### Q3：每个 generator 跑 5 条 audit 够吗？

不够作为统计置信。5 条可以证明工程链路基本通；正式扩大前建议每个 generator 跑 50-100 条 seed audit。

### Q4：`difficulty_mix` 已经有 60/30/10，是否代表真实规模难度已生效？

当前还不能这样理解。因为所有 profile 的 `difficulty_support` 当前都是 `basic`，系统会记录 difficulty label，但大多数 generator 尚未把难度映射成真实规模参数。

### Q5：LLM 临时失败会不会直接丢样本？

Streaming 模式下，timeout、网络错误、HTTP 5xx 等瞬时错误会 requeue。业务质量错误不会 requeue。

### Q6：forward eval 失败最多说明什么？

它说明 LLM 从 NL 重新建模后的代码没有通过 solver truth 验证。常见原因是漏固定成本、漏流平衡、模型 infeasible、目标方向或约束结构错误。应优先查看：

```text
08_forward_eval/rejected_pairs.jsonl
08_forward_eval/forward_eval_rejection_dashboard.md
reports/run_quality_dashboard.md
```

### Q7：改 prompt 后能否复用已有 seed？

可以。推荐复用 Seed Factory 输出的：

```text
04b_instance_quality/quality_validated_instances.jsonl
```

重新运行 `run-cpt-from-seeds`，这样可以专门评估 prompt 变化对 05-10 的影响。

## 14. 当前边界与下一步

当前已经具备：

- 53 个 enabled generator profile。
- 24 个任务族的 scenario catalog 覆盖。
- 13 个 family contract 和 generator overrides。
- Seed audit、Seed Factory、CPT Factory 三类主工作流。
- Gurobi reference truth。
- 04b seed 质量门。
- NL 质量过滤。
- forward modeling 静态检查和 family contract 检查。
- objective match 验收。
- forward repair 单轮修复。
- train-only streaming export。
- run quality dashboard、diversity dashboard、generator quality report、production plan。

当前仍需持续增强：

- 将更多 generator 的 `difficulty_support` 从 `basic` 升级为真实 native difficulty。
- 增强 forward eval 高频失败家族的 prompt 和 family contract。
- 提升 forward repair 恢复率。
- 增加 embedding / MinHash 级近重复检查和跨 run 去重。
- 在正式评估需要时重新设计 val/test split，避免与 train 在 generator、scenario 或 seed 上泄漏。

## 15. 推荐操作顺序总结

如果你是第一次接手当前 OR-CPT Engine，建议按下面顺序跑：

```text
1. scan-generators
2. smoke-test-generators
3. audit-generators --samples-per-generator 50
4. analyze-generator-quality
5. plan-production --target-accepted 10000
6. run-seed-factory --production-plan ... --num-instances 1000
7. review quality_validated_instances.jsonl 和 seed_manifest
8. run-cpt-from-seeds --pipeline-mode streaming
9. review train.jsonl、cpt_run_manifest、run_quality_dashboard、diversity_dashboard
10. 根据 rejected 和 dashboard 调整 prompt / contract / generator profile
```

最重要的生产原则：

```text
先把 seed 做干净，再让 LLM 花钱；
LLM 可以生成，但最终必须让 Gurobi 验收；
每次生产都要看 manifest 和 dashboard，不只看 train.jsonl 行数。
```
