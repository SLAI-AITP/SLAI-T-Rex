# OR-CPT 配置说明

本目录包含 `or_cpt_engine` 的所有配置文件。

---

## 配置清单

| 文件 | 用途 |
|---|---|
| `project.yaml` | 项目级元信息（名称、版本等） |
| `paths.yaml` | 路径配置（外部依赖、输出目录等） |
| `llm.yaml` | LLM 模型接入配置（API 池、并发、各阶段参数） |
| `generation_plan.yaml` | 生成计划（每 generator 产出量、难度混合比例、过滤参数） |
| `generator_profiles.yaml` | Generator 档案库（见下方详细说明） |
| `quality_thresholds.yaml` | 质量阈值（NL 过滤、前向评估容差等） |
| `cpt_rendering.yaml` | CPT 渲染配置（文档类型比例等） |
| `train_val_split.yaml` | 历史兼容配置；当前 train-only export 不再使用 |

---

## `generator_profiles.yaml` — Generator 档案库

### 定位

这是 OR-CPT 数据工厂的 **generator 档案库**，为每个 OptMATH generator 定义了一份完整的"身份档案"（约 50+ 个，文件约 5000 行）。它是后续做精细化生产调度（按领域分层采样、按难度混合、按质量预期监控）的数据基础。

### 每个 profile 包含的字段

| 字段 | 用途 |
|---|---|
| `enabled` / `priority` / `recommended_weight` | **生产调度**：哪些 generator 能用、优先级高低（`high`/`medium`/`cautious`）、权重多少 |
| `sampling_weight` | 控制每个 generator 的产量：`cpt_weight`（采样权重）、`max_daily_samples`（日产量上限）、`min_quality_score`（最低质量门槛）、`solver_timeout_seconds`（求解超时） |
| `task_family` / `sub_family` | 问题分类（如 `facility_location`、`routing`、`scheduling`、`diet`、`packing` 等） |
| `modeling_concepts` | 该 generator 覆盖的数学概念（如 `binary_assignment`、`capacity_constraint`、`flow_balance`） |
| `canonical_math_signature` | 该问题类型的"标准数学签名"：应有变量类型、核心约束、必要数据表、**禁止改写的事项**（`forbidden_changes`） |
| `difficulty_config` / `difficulty_axes` | 4 个难度等级的配置（`level_1` ~ `level_4`），部分有具体数值范围（如 `facility_count`、`customer_count`、`period_count`、`node_count`） |
| `quality_risks` | LLM 在该 generator 上的已知常见失败模式（如 "LLM may omit fixed opening costs"） |
| `rationale_requirements` | LLM 在生成 `modeling_rationale`（建模推理文档）时应当解释的关键内容 |
| `quality_controls` | 验收要求清单（`validation_required`）+ 拒绝原因分类（`rejection_reasons`） |
| `max_views_per_seed_instance` | 每个实例最多生成多少种视角/变体 |
| `expected_acceptance_rate` | 预期的全链路通过率 |
| `formulation_variants` | 建模变体列表（当前均只有一个 `base` variant） |
| `semantic_scenarios` | 语义场景（预留，当前均为空） |

### 与流水线的关系

`run-full` 在阶段 03（`instance_generation`）读取此文件，当前已生效的能力：

1. **`enabled: false` 的 generator 会被跳过**，不生成实例
2. **`difficulty_mix` 概率混采**：通过 `generation_plan.yaml` 中的 `difficulty_mix` 配置，配合 `select_difficulty_level()` 做确定性的难度等级抽样（SHA-1 哈希 → 权重轮盘）
3. **Profile 元信息**（`sub_family`、`canonical_math_signature`、`concept_tags`、`variant_id` 等）会写入每个 `InstanceRecord`，沿整条流水线传递

### `difficulty_support` 说明

每个 profile 的 `difficulty_support` 字段决定难度参数是否实际传递给 OptMATH generator：

| 值 | 行为 |
|---|---|
| `"basic"`（当前所有 profile 的默认值） | 只记录 `difficulty_level` 标签，不传递规模/约束参数给 generator |
| `"native"` | 从 `difficulty_config` 轴的对应 level 提取具体参数（如 `size_scale: small`、`facility_count: [3,5]`），传给 OptMATH generator 控制实际生成规模 |
| `"none"` | 难度标签也无意义，始终 `level_1` |

当前所有 profile 均为 `basic`，难度混采的概率分布机制已跑通但规模参数尚未实际生效。后续将部分 generator 升级为 `"native"` 后，profile 中的具体难度轴参数才会控制生成实例的规模。

---

## `generation_plan.yaml` — 生成计划

控制阶段 03 的核心参数：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `run_name` | `or_cpt_mvp` | 自动生成 run_id 时使用的前缀（`--run-id` 可覆盖） |
| `source.type` | `optmath_generators` | 数据源类型（预留） |
| `generation.num_instances_per_generator` | `5` | 每个 generator 生成实例数（`--num-instances` 可覆盖） |
| `generation.max_iter_per_generator` | `200` | 最大迭代次数（预留，当前未实现） |
| `generation.num_workers` | `4` | 并行 worker 数（预留，当前未实现） |
| `filters.var_num_max` | `200` | 变量数上限（预留，当前未实现） |
| `filters.constraint_num_max` | `500` | 约束数上限（预留，当前未实现） |
| `filters.solve_timeout_sec` | `60` | Gurobi 求解超时秒数（阶段 04 和 08 生效） |
| `difficulty_mix.level_1/2/3` | `0.60/0.30/0.10` | 难度等级混采比例（阶段 03 生效） |

---

## `train_val_split.yaml` — 历史兼容配置

当前 OR-CPT 生产系统只输出 `train.jsonl`，不再做 train / val / test 切分。

这个文件暂时保留，是为了兼容 `_load_config` 和历史 CLI 参数；阶段 10 的 train-only export 不会读取其中的比例参数。后续如果确认没有旧脚本依赖，可以再做一次配置清理，把 `train_val_split.yaml` 和相关 CLI 参数一并删除。

---

## `llm.yaml` — LLM 配置

| 节 | 说明 |
|---|---|
| `providers.default` | 默认 LLM provider（API 池路径、模型名、超时、重试、并发、端点健康管理） |
| `backtranslation` | 反向翻译阶段参数（temperature、max_tokens、候选数） |
| `forward_modeling` | 正向建模阶段参数（temperature、max_tokens、候选数） |
| `rendering` | 渲染阶段参数（预留，当前未使用 LLM） |

---

*最后更新：2026-06-05*
