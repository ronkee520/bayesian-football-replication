# Bayesian Football Model Replication

[中文说明](#中文说明) · [English Guide](#english-guide)

This repository reproduces and extends the Bayesian hierarchical football models
in Baio and Blangiardo (2010) using the 2023/24 English Premier League season.

---

## 中文说明

### 1. 项目简介

本项目使用层次贝叶斯模型分析 2023/24 赛季英超联赛的 380 场比赛，估计：

- 每支球队相对于联赛平均水平的进攻能力；
- 每支球队对对手进球率的防守影响；
- 英超整体主场优势；
- 比赛比分、胜平负概率和赛季场上积分的后验预测分布。

项目复现并扩展以下论文：

> Baio, G. and Blangiardo, M. (2010). Bayesian hierarchical model for the
> prediction of football results. *Journal of Applied Statistics*, 37(2),
> 253–264. <https://doi.org/10.1080/02664760802684177>

项目研究三个问题：

1. 基础层次泊松模型能否得到可解释的进攻、防守和主场优势参数？
2. 三成分重尾混合模型是否改变对实力极端球队的估计？
3. 样本内拟合的改善能否延续到最后若干轮的样本外预测？

### 2. 模型概览

对于比赛 $g$，主队和客队进球分别建模为：

$$
y_{g,h}\sim\mathrm{Poisson}(\theta_{g,h}),\qquad
y_{g,a}\sim\mathrm{Poisson}(\theta_{g,a}).
$$

基础模型使用：

$$
\log\theta_{g,h}
=\text{home}+\text{attack}_{h(g)}+\text{defence}_{a(g)},
$$

$$
\log\theta_{g,a}
=\text{attack}_{a(g)}+\text{defence}_{h(g)}.
$$

其中，$h(g)$ 和 $a(g)$ 分别表示比赛 $g$ 的主队和客队；
$\text{home}$ 是整体主场优势；$\text{attack}_t$ 与
$\text{defence}_t$ 分别是球队 $t$ 相对于联赛平均水平的进攻和防守效应。

混合模型将球队效应的单一总体分布替换为弱、中、强三个 Student-t 成分。
离散分组变量已经边缘化，因此可以直接使用 NumPyro NUTS 对连续后验进行采样。

项目保留两套相互独立的先验配置：

- `paper_replication`：尽量贴近论文及 BUGS 附录，用于回答论文复现问题；
- `modernized`：使用非中心化参数和正则化尺度先验，用于比较现代参数化的影响。

### 3. 项目结构

```text
bayesian-football-replication/
├── configs/                 # 模型、先验、随机种子和采样器配置
├── data/
│   ├── raw/                 # 不可覆盖的原始比赛数据
│   └── processed/           # 脚本生成的确定性派生数据
├── docs/                    # 中文方法、复现差异和验证记录
├── notebooks/               # 可选的展示型 Notebook
├── results/                 # MCMC、诊断、预测表和图表输出
├── scripts/                 # 数据验证、正式拟合和留出预测入口
├── src/bayes_football/      # 模型与分析工作流的核心实现
├── tests/                   # 数据、数学逻辑、预测和 MCMC 测试
├── pyproject.toml           # Python 项目与依赖声明
└── uv.lock                  # 锁定的可复现依赖环境
```

重要文档：

- [`docs/methods_zh.md`](docs/methods_zh.md)：层次贝叶斯、混合模型、NUTS 和预测评估讲义；
- [`docs/replication_notes.md`](docs/replication_notes.md)：论文正文与附录差异及代码选择；
- [`docs/reproducibility.md`](docs/reproducibility.md)：正式结果的验收标准；
- [`docs/verification.md`](docs/verification.md)：已经完成的测试和端到端运行记录。

### 4. 在 VS Code 中完整复现

以下步骤以项目根目录 `bayesian-football-replication` 为起点。项目通过 `uv.lock`
固定依赖版本；Windows、macOS 和 Linux 均可执行同一组 `uv run` 命令。

#### 4.1 安装准备

请先安装：

- Git；
- Visual Studio Code；
- Python 3.10–3.12；
- VS Code 的 Microsoft Python 扩展；
- 可选：Jupyter 扩展，用于以后查看 Notebook。

如果尚未安装 `uv`，可在系统终端运行：

```bash
python -m pip install uv
```

#### 4.2 在 VS Code 中打开正确文件夹

1. 启动 VS Code。
2. 选择 `File → Open Folder`。
3. 打开包含 `README.md`、`pyproject.toml` 和 `uv.lock` 的仓库根目录。
4. 使用 ``Ctrl+` `` 打开 VS Code 集成终端。
5. 确认终端当前位置包含 `README.md`、`pyproject.toml` 和 `uv.lock`。

可用下面的命令检查：

```bash
git status
```

正常情况下应显示当前分支为 `main`，且没有未提交修改。

#### 4.3 根据锁文件创建独立环境

在 VS Code 集成终端运行：

```bash
uv sync --extra dev --frozen
```

该命令会在项目内创建 `.venv`，并严格按照 `uv.lock` 安装 JAX、NumPyro、ArviZ、
Pandas、Pytest 等依赖。`--frozen` 可以防止复现时意外重新解析依赖版本。

然后在 VS Code 中按 `Ctrl+Shift+P`，执行 `Python: Select Interpreter`，选择：

- Windows：`.venv\Scripts\python.exe`
- macOS/Linux：`.venv/bin/python`

选择后重新打开一个集成终端。也可以始终使用下面的 `uv run` 命令，这样无需手动
激活虚拟环境。

检查核心依赖：

```bash
uv run python -c "import jax, numpyro, arviz; print(jax.__version__, numpyro.__version__, arviz.__version__)"
```

#### 4.4 验证原始数据

运行：

```bash
uv run python scripts/validate_data.py
```

正确输出应包含：

```text
Validated 380 matches and 20 teams.
Observed on-field table written to data\processed\observed_on_field_table.csv.
```

数据控制结果应满足：

- 380 场比赛；
- 20 支球队；
- 每队 38 场，其中主场 19 场、客场 19 场；
- 没有重复的有序主客场对阵；
- 曼城的场上积分为 91 分；
- 数据文件 SHA-256 为
  `1b405d19e804f5221d65472ffc3fa0ca9044ba957c9a83775ca41bb47dab2dc0`。

这里的积分是根据比赛结果计算的**扣分前场上积分**，不包含 Everton 和
Nottingham Forest 的行政扣分。

#### 4.5 运行自动测试

先运行不包含 MCMC 的快速测试：

```bash
uv run pytest -m "not slow"
```

当前版本的正确结果是 `10 passed, 4 deselected`。

随后运行包括四个微型 NUTS 拟合在内的全部测试：

```bash
uv run pytest
```

当前版本的正确结果是 `14 passed`。第一次运行 JAX 时需要编译，耗时会比后续运行长。

再执行代码质量检查：

```bash
uv run ruff check .
```

正确结果为 `All checks passed!`。

#### 4.6 先运行快速端到端检查

快速配置只用于确认完整流水线可以工作，不用于统计结论：

```bash
uv run python scripts/run_model.py --config configs/quick_smoke.yaml
uv run python scripts/run_model.py --config configs/mixture_quick_smoke.yaml
uv run python scripts/run_holdout.py --config configs/holdout_quick_smoke.yaml
```

三个命令应分别在以下目录生成结果：

```text
results/quick_smoke/
results/mixture_quick_smoke/
results/holdout_quick_smoke/
```

快速配置只有一条短链，因此 R-hat 会显示为空或不可计算，这是预期现象。

#### 4.7 运行正式论文复现

运行论文风格的基础模型和混合模型：

```bash
uv run python scripts/run_model.py --config configs/basic_paper_replication.yaml
uv run python scripts/run_model.py --config configs/mixture_paper_replication.yaml
```

这两组结果用于讨论论文模型本身，包括宽先验、混合分布和计算困难。

#### 4.8 运行现代化配置

运行现代化基础模型：

```bash
uv run python scripts/run_model.py --config configs/basic_modernized.yaml
```

运行边缘化 Student-t 混合模型：

```bash
uv run python scripts/run_model.py --config configs/mixture_modernized.yaml
```

正式配置使用四条链。混合模型计算量较大，在纯 CPU 环境中可能需要较长时间；
采样完成前需保持终端进程运行。

#### 4.9 运行最后六轮留出预测

```bash
uv run python scripts/run_holdout.py --config configs/holdout_modernized.yaml
```

该实验使用前 320 场拟合，并把文件最后 60 场作为测试集。由于原始数据没有日期和
轮次字段，“每 10 行构成一轮”是明确记录的近似假设。

#### 4.10 检查正式输出

每次全赛季模型运行会生成：

| 文件 | 内容 |
| --- | --- |
| `config_resolved.yaml` | 实际使用的模型、种子和采样器配置 |
| `posterior_samples.npz` | 按链保存的后验样本 |
| `diagnostics.csv` | rank-normalized R-hat、bulk ESS 和 tail ESS |
| `sampler_diagnostics.json` | divergence、接受率、步数和 BFMI |
| `posterior_predictive_table.csv` | 各队场上积分的后验预测与区间 |

留出实验还会生成：

- `heldout_match_predictions.csv`：60 场测试比赛的进球均值和胜平负概率；
- `heldout_metrics.json`：进球 MAE、三分类 Brier score、命中率和比分对数预测密度。

正式结果应至少满足以下检查：

- divergence 数量为 0；
- rank-normalized R-hat 通常不超过 1.01；超过时结合轨迹图与多峰性进一步检查；
- 所有用于结论的参数具有足够的 bulk ESS 与 tail ESS；
- 每条链的 BFMI 没有明显异常；
- 论文复现结果和现代化结果使用相同数据后再比较；
- 样本内 posterior predictive check 与样本外预测不能混称为“预测准确率”。

不同 CPU、JAX 版本和浮点实现可能带来很小的数值差异。正确复现强调后验分布、
诊断结论和预测指标在 Monte Carlo 误差范围内一致，而不是要求每一个小数完全相同。

#### 4.11 生成汇总表和可视化结果

完成至少一个正式的全赛季模型拟合后，运行：

```bash
uv run python scripts/summarize_results.py
```

该命令默认自动发现完整结果，并排除所有 `quick_smoke` 目录，以免把流程测试的小样本
当作正式研究结论。完成后首先打开：

```text
results/summary.md
```

主要输出包括：

| 文件 | 内容 |
| --- | --- |
| `results/tables/model_comparison.csv` | MAE、RMSE、预测区间覆盖率、divergence、R-hat、ESS 和 BFMI |
| `results/tables/posterior_parameter_summary.csv` | 主场优势、模型超参数以及各球队攻防能力的后验摘要 |
| `results/diagnostics/convergence_summary.csv` | 可解释参数的逐项收敛诊断及自动检查状态 |
| `results/figures/model_comparison.png` | 预测表现与采样质量的综合比较 |
| `results/figures/observed_vs_predicted_points.png` | 实际积分、预测均值及 90% 预测区间 |
| `results/figures/team_prediction_errors.png` | 各球队预测误差及模型间比较 |
| `results/figures/convergence_overview.png` | 两个正式模型的四链 MCMC 收敛总览 |
| `results/figures/attack_effects_*.png` | 球队进攻效应及 90% 后验区间 |
| `results/figures/defence_effects_*.png` | 球队防守效应及 90% 后验区间 |
| `results/figures/trace_*.png` | 关键参数和代表性球队的 MCMC 轨迹图 |
| `results/figures/rank_*.png` | 不同链是否均匀探索后验分布的 rank plot |

如需明确指定要比较的运行目录，可使用：

```bash
uv run python scripts/summarize_results.py --runs results/basic_paper_replication results/mixture_paper_replication
```

`trace` 和 `rank` 图只选择主场优势、总体尺度或混合成分以及强弱代表球队；逐场期望
进球等数百个派生参数仍保存在数据文件中，但不会全部绘图。轨迹图需要和 R-hat、
ESS、divergence 及 BFMI 一起判断，不能单独作为收敛证明。

#### 4.12 从全新克隆到完整结果的最短路径

其他人在 GitHub 克隆本仓库并进入项目根目录后，可以按顺序直接运行：

```bash
uv sync --extra dev --frozen
uv run python scripts/validate_data.py
uv run pytest
uv run python scripts/run_model.py --config configs/basic_paper_replication.yaml
uv run python scripts/run_model.py --config configs/mixture_paper_replication.yaml
uv run python scripts/summarize_results.py
```

最后一条命令在两次正式拟合完成后执行。成功时终端会显示
`Generated 16 summary files from 2 fitted runs.`。随后打开 `results/summary.md`，并确认
`results/figures/` 中存在 `convergence_overview.png`、两个 `trace_*.png` 和两个
`rank_*.png`。`quick_smoke` 只有一条链，只能验证代码能够运行，不能生成有意义的
多链收敛判断。

仓库不会预先附带上述结果。完成命令后，用户会在自己的电脑上得到以下多链诊断图：

```text
results/figures/convergence_overview.png
results/figures/trace_basic_paper_replication.png
results/figures/trace_mixture_paper_replication.png
results/figures/rank_basic_paper_replication.png
results/figures/rank_mixture_paper_replication.png
```

其中，trace plot 使用不同颜色显示四条独立 MCMC 链；rank plot 用于检查各条链是否
均匀探索同一个后验分布。这些文件均由本地后验样本即时生成，不是仓库内预置图片。

### 5. 常见问题

#### VS Code 找不到模块

重新执行 `Python: Select Interpreter` 并选择项目的 `.venv`。终端命令使用
`uv run python ...`，以确保调用锁定环境中的解释器。

#### JAX 首次运行较慢

第一次拟合需要即时编译。这不是程序卡死。快速测试通过后，再运行正式四链配置。

#### 内存或运行时间不足

资源受限时，可复制一个 YAML 配置并减少 `warmup`、`samples` 或 `chains` 进行
调试。正式分析仍使用四链配置，并重新检查采样诊断。

#### Windows 终端无法激活 `.venv`

无需改变 PowerShell 执行策略，直接使用 `uv run` 即可。若使用传统虚拟环境，
Windows 激活命令为：

```powershell
.venv\Scripts\Activate.ps1
```

---

## English Guide

### 1. Project overview

This project applies hierarchical Bayesian models to all 380 matches in the
2023/24 English Premier League season. It estimates team attack effects, team
defence effects, the common home advantage, score distributions, match-outcome
probabilities, and posterior distributions for on-field league points.

The project asks whether a basic hierarchical Poisson model produces
interpretable team effects, whether a heavy-tailed three-component mixture
better represents unusually strong or weak teams, and whether an in-sample
improvement survives a time-ordered holdout experiment.

The two prior profiles have different purposes:

- `paper_replication` follows the published paper and BUGS appendix as closely
  as practical;
- `modernized` uses non-centred effects and regularising scale priors to assess
  the effect of a contemporary parameterisation.

### 2. Repository structure

```text
configs/                 Model, prior, seed, and sampler configurations
data/raw/                Immutable match-level source data
data/processed/          Deterministic generated data
docs/                    Method, replication, and reproducibility notes
notebooks/               Optional presentation notebooks
results/                 Generated posterior, diagnostic, and prediction files
scripts/                 Data validation, model fitting, and holdout entry points
src/bayes_football/      Core implementation
tests/                   Data, mathematical, prediction, and MCMC tests
pyproject.toml           Python package and dependency declaration
uv.lock                  Locked reproducible environment
```

### 3. Full reproduction in VS Code

#### 3.1 Prerequisites

Install Git, Visual Studio Code, Python 3.10–3.12, the Microsoft Python extension,
and optionally the Jupyter extension.

Install `uv` if it is not already available:

```bash
python -m pip install uv
```

#### 3.2 Open the repository

In VS Code, select `File → Open Folder` and open the
`bayesian-football-replication` directory itself. Open the integrated terminal
with ``Ctrl+` `` and confirm that `pyproject.toml` and `uv.lock` are present.

#### 3.3 Recreate the locked environment

```bash
uv sync --extra dev --frozen
```

Run `Python: Select Interpreter` from the Command Palette and select
`.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on macOS/Linux.
All commands below use `uv run`, so manual environment activation is optional.

Verify the core libraries:

```bash
uv run python -c "import jax, numpyro, arviz; print(jax.__version__, numpyro.__version__, arviz.__version__)"
```

#### 3.4 Validate the data

```bash
uv run python scripts/validate_data.py
```

The expected output reports 380 matches and 20 teams. The generated on-field table
gives Manchester City 91 points. These are points earned from match results
before administrative deductions.

#### 3.5 Run tests and linting

```bash
uv run pytest -m "not slow"
uv run pytest
uv run ruff check .
```

For the current version, the expected results are 10 fast tests passed, 14 total
tests passed, and `All checks passed!` from Ruff.

#### 3.6 Run end-to-end smoke checks

```bash
uv run python scripts/run_model.py --config configs/quick_smoke.yaml
uv run python scripts/run_model.py --config configs/mixture_quick_smoke.yaml
uv run python scripts/run_holdout.py --config configs/holdout_quick_smoke.yaml
```

Smoke configurations use one short chain and only establish that the complete
pipeline works. R-hat is unavailable, and these numerical estimates are not used
for substantive analysis.

#### 3.7 Run the full paper-style replication

```bash
uv run python scripts/run_model.py --config configs/basic_paper_replication.yaml
uv run python scripts/run_model.py --config configs/mixture_paper_replication.yaml
```

#### 3.8 Run the modernized specifications

```bash
uv run python scripts/run_model.py --config configs/basic_modernized.yaml
uv run python scripts/run_model.py --config configs/mixture_modernized.yaml
uv run python scripts/run_holdout.py --config configs/holdout_modernized.yaml
```

The full configurations use four chains. The marginalized mixture model can take
substantial time on a CPU-only machine. Keep the VS Code terminal running until
sampling and diagnostic export have finished.

#### 3.9 Verify the outputs

Full-season fits write the resolved configuration, posterior sample archive,
ArviZ diagnostics, sampler diagnostics, and posterior predictive league table
to the configured directory under `results/`. The holdout run additionally
writes match-level predictions and a JSON metric summary.

Before reporting a result, verify:

- zero divergent transitions;
- rank-normalized R-hat close to 1 and normally no greater than 1.01;
- adequate bulk and tail effective sample sizes;
- acceptable BFMI for every chain;
- clear separation between in-sample posterior predictive checks and held-out
  forecast performance.

Small numerical differences across CPUs and JAX versions are normal. A correct
reproduction requires substantively equivalent posterior distributions,
diagnostics, and predictive metrics within Monte Carlo error; it does not require
identical last decimal places.

#### 3.10 Generate summary tables and figures

After at least one completed full-season fit, run:

```bash
uv run python scripts/summarize_results.py
```

The command automatically discovers complete runs and excludes `quick_smoke`
directories by default. Open `results/summary.md` first. It indexes the compact
model-comparison table, posterior parameter summaries, convergence diagnostics,
observed-versus-predicted plot, team error plot, attack/defence interval plots,
and selected MCMC trace and rank plots.

To select runs explicitly:

```bash
uv run python scripts/summarize_results.py --runs results/basic_paper_replication results/mixture_paper_replication
```

The trace and rank figures deliberately show a small set of interpretable latent
parameters rather than hundreds of match-level deterministic quantities. Visual
inspection is interpreted together with R-hat, ESS, divergences, and BFMI.

#### 3.11 Shortest clean-clone reproduction path

After cloning the repository and entering its root directory, run:

```bash
uv sync --extra dev --frozen
uv run python scripts/validate_data.py
uv run pytest
uv run python scripts/run_model.py --config configs/basic_paper_replication.yaml
uv run python scripts/run_model.py --config configs/mixture_paper_replication.yaml
uv run python scripts/summarize_results.py
```

The last command runs after both four-chain fits have completed. It should
report `Generated 16 summary files from 2 fitted runs.`. Open
`results/summary.md`, then verify that `results/figures/` contains the compact
convergence overview, two detailed trace plots, and two rank plots. Single-chain
smoke runs only validate the pipeline and cannot establish multi-chain convergence.

The repository intentionally ships without generated results. After running the
commands, each user locally creates `convergence_overview.png`, the two detailed
`trace_*.png` files, and the two `rank_*.png` files under `results/figures/`.
Different trace colors represent the four independent MCMC chains; none of these
images are pre-populated in the repository.

### 4. Reference

Baio, G. and Blangiardo, M. (2010). Bayesian hierarchical model for the
prediction of football results. *Journal of Applied Statistics*, 37(2),
253–264. <https://doi.org/10.1080/02664760802684177>
