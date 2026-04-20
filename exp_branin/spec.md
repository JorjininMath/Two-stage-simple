# spec.md — exp_branin 实验规格说明

## 1. 实验目标

**核心问题**：CKME-CP 在二维输入、不同噪声形态（Gaussian vs. 异方差重尾 Student-t）下，相比 DCP-DR 和 hetGP，能否以更窄的区间实现更接近标称覆盖率（90%）的条件覆盖？

具体地：
- Gaussian 噪声场景作为**基线**（所有方法应表现相近，用于验证实现正确性）。
- Student-t 噪声场景中 df 和 scale 均依赖第一维输入，造成**空间变化的尾部形态**，这是 CKME 相对 hetGP（假设 Gaussian 条件分布）的优势区域，也是相对 DCP-DR 的差异区域。
- 使用 Branin-Hoo 作为真实函数，因为它已有标准基准实现，且在同一代码库中已用于其他实验，便于比较。

---

## 2. Simulator 数学规格

### 2.1 Branin-Hoo 真实函数

沿用 `Two_stage/sim_functions/exp3.py` 中的实现（`exp3_true_function`）：

```
f(x) = a*(x2 - b*x1^2 + c*x1 - r)^2 + s*(1-t)*cos(x1) + s

参数（标准文献值）：
  a = 1.0
  b = 5.1 / (4*pi^2)
  c = 5 / pi
  r = 6.0
  s = 10.0
  t = 1 / (8*pi)
```

**输入域**：`x1 ∈ [-5, 10]`,  `x2 ∈ [0, 15]`（即 `EXP3_X_BOUNDS`）

### 2.2 共享的噪声参数函数

第一维原始值 `x1 ∈ [-5, 10]` 先归一化为：
```
x1_scaled = (x1 - (-5)) / (10 - (-5)) ∈ [0, 1]
```

**噪声 scale 函数**（两种噪声共用）：
```
sigma(x) = 0.4 * (4 * x1_scaled + 1)
```
- 当 x1=-5 时 sigma=0.4；当 x1=10 时 sigma=2.0（线性增长，异方差性较强）

**噪声自由度函数**（仅 Student-t 使用）：
```
nu(x) = max(2.0,  6 - 4 * x1_scaled)
```
- 当 x1=-5 时 nu=6；当 x1=10 时 nu=2（极重尾）

### 2.3 场景 A：`branin_gauss`（Gaussian 噪声）

```
Y = f(x) + epsilon,    epsilon ~ N(0, sigma(x)^2)
sigma(x) = 0.4 * (4 * x1_scaled + 1)
```

### 2.4 场景 B：`branin_student`（Student-t 噪声）

```
Y = f(x) + epsilon,    epsilon ~ t_{nu(x)}(0, sigma(x))
nu(x)    = max(2.0, 6 - 4 * x1_scaled)
sigma(x) = 0.4 * (4 * x1_scaled + 1)
```

**采样方法**（向量化，无需逐点循环）：
```
rng = np.random.default_rng(random_state)
z = rng.standard_normal(n)               # N(0,1)
v = rng.chisquare(nu_arr) / nu_arr       # chi2(nu)/nu  [numpy支持array-valued df]
eps = z / np.sqrt(v) * sigma_arr        # t_{nu_i}(0, sigma_i)
```

> **注意**：nu=2 时 Student-t 方差理论上无穷大，样本中偶尔出现极大值属正常。
> 依赖 percentile-based t_grid（见 §3.4）防止极端值破坏 CDF grid。

---

## 3. Pipeline 配置

### 3.1 config.txt 参数

```
simulator_func = branin_gauss   # 占位；run 脚本中按 case 覆盖
n_0 = 300
r_0 = 10
n_1 = 500
r_1 = 5
n_test = 500
r_test = 1
n_cand = 2000
t_grid_size = 500
ell_x = 1.0
lam = 1e-2
h = 0.3
alpha = 0.1
```

**设计说明**：
- `n_0=300, r_0=10`：Stage 1 共 3000 个样本。2D 域相比 1D 需要更多 sites 覆盖空间。
- `n_1=500, r_1=5`：Stage 2 共 2500 个样本。
- `n_cand=2000`：候选点集，2D 域需要更密集覆盖。
- `t_grid_size=500`：与 exp_nongauss 一致，Student-t 重尾需要足够宽的 grid。
- `alpha=0.1`：90% 标称覆盖。

### 3.2 Stage 2 方法

默认 `method=lhs`，与 exp_nongauss 默认一致。支持 `--method mixed` CLI 选项。

### 3.3 超参数调优策略

与 exp_nongauss 完全相同的流程：

1. 先运行 `exp_branin/pretrain_params.py` 对两个 simulator 分别做 k-fold CV。
2. 结果保存到 `exp_branin/pretrained_params.json`，格式与 `exp_nongauss/pretrained_params.json` 相同。
3. main experiment script 自动加载；若 JSON 不存在则 fallback 到 config.txt 默认参数并打印警告。

**CV 搜索网格**（2D 域调整，ell_x 需更大）：
```python
ParamGrid(
    ell_x_list=[0.5, 1.0, 2.0, 4.0],   # 2D 空间更大，需更大 ell_x
    lam_list=[1e-3, 1e-2, 1e-1],
    h_list=[0.1, 0.3, 0.5],
)
```

Pilot 规模：`n_pilot=150, r_pilot=5, cv_folds=5`。

### 3.4 t_grid 构建

严格遵循 percentile-based 规则（CLAUDE.md）：
```python
Y_lo = np.percentile(Y_all, 0.5)
Y_hi = np.percentile(Y_all, 99.5)
y_margin = 0.10 * (Y_hi - Y_lo)
t_grid = np.linspace(Y_lo - y_margin, Y_hi + y_margin, t_grid_size)
```
对 Student-t 噪声（特别是 nu→2 时 scale 很大），此规则尤为重要。

---

## 4. 文件结构

### 4.1 新建文件列表

```
Two_stage/sim_functions/sim_branin_gauss.py       # Gaussian 噪声 simulator
Two_stage/sim_functions/sim_branin_student.py     # Student-t 噪声 simulator

exp_branin/
    spec.md                       # 本文件
    config.txt                    # 实验参数
    config_utils.py               # 1行 import
    pretrain_params.py            # CV 调参脚本
    run_branin_compare.py         # 主实验脚本
    plot_branin.py                # 覆盖率/宽度/IS 图
    pretrained_params.json        # 由 pretrain_params.py 生成（运行后产生）
```

### 4.2 修改文件

```
Two_stage/sim_functions/__init__.py   # 新增 2 条 import + 2 条 registry 条目
```

---

## 5. Simulator 文件详细规格

### 5.1 `Two_stage/sim_functions/sim_branin_gauss.py`

```python
"""
DGP branin_gauss: Branin-Hoo true function + heteroscedastic Gaussian noise.
  Y = f(x) + epsilon,
  epsilon ~ N(0, sigma(x)^2)
  sigma(x) = 0.4 * (4 * x1_scaled + 1)
  x1_scaled = (x1 - (-5)) / (10 - (-5)),  x1 = x[:, 0]

Domain: x1 in [-5, 10],  x2 in [0, 15]  (EXP3_X_BOUNDS)
"""

import numpy as np
from .exp3 import exp3_true_function, EXP3_X_BOUNDS

BRANIN_GAUSS_X_BOUNDS = EXP3_X_BOUNDS

def _x1_scaled(x: np.ndarray) -> np.ndarray:
    x_2d = np.atleast_2d(np.asarray(x, dtype=float))
    return (x_2d[:, 0] - (-5.0)) / (10.0 - (-5.0))

def _sigma(x: np.ndarray) -> np.ndarray:
    return 0.4 * (4.0 * _x1_scaled(x) + 1.0)

def branin_gauss_simulator(x, random_state=None):
    rng = np.random.default_rng(random_state)
    x_2d = np.atleast_2d(np.asarray(x, dtype=float))
    sigma = _sigma(x_2d)
    return exp3_true_function(x_2d) + rng.normal(0.0, sigma)
```

### 5.2 `Two_stage/sim_functions/sim_branin_student.py`

```python
"""
DGP branin_student: Branin-Hoo true function + heteroscedastic Student-t noise.
  Y = f(x) + epsilon,
  epsilon ~ t_{nu(x)}(0, sigma(x))
  nu(x)    = max(2.0, 6 - 4 * x1_scaled)   [ranges from 6 at x1=-5 to 2 at x1=10]
  sigma(x) = 0.4 * (4 * x1_scaled + 1)

Note: nu(x) can reach 2.0 where Student-t variance is undefined (infinite).
Use percentile-based t_grid bounds (not min/max) to handle extreme outliers.

Domain: x1 in [-5, 10],  x2 in [0, 15]  (EXP3_X_BOUNDS)
"""

import numpy as np
from .exp3 import exp3_true_function, EXP3_X_BOUNDS

BRANIN_STUDENT_X_BOUNDS = EXP3_X_BOUNDS

def _x1_scaled(x: np.ndarray) -> np.ndarray:
    x_2d = np.atleast_2d(np.asarray(x, dtype=float))
    return (x_2d[:, 0] - (-5.0)) / (10.0 - (-5.0))

def _sigma(x: np.ndarray) -> np.ndarray:
    return 0.4 * (4.0 * _x1_scaled(x) + 1.0)

def _nu(x: np.ndarray) -> np.ndarray:
    return np.maximum(2.0, 6.0 - 4.0 * _x1_scaled(x))

def branin_student_simulator(x, random_state=None):
    rng = np.random.default_rng(random_state)
    x_2d = np.atleast_2d(np.asarray(x, dtype=float))
    n = x_2d.shape[0]
    nu_arr = _nu(x_2d)      # shape (n,)
    sigma_arr = _sigma(x_2d)  # shape (n,)
    # Vectorized t_{nu}(0, sigma) via z/sqrt(chi2(nu)/nu)*sigma
    z = rng.standard_normal(n)
    v = rng.chisquare(nu_arr) / nu_arr   # numpy supports array-valued df (>= 1.17)
    eps = z / np.sqrt(v) * sigma_arr
    return exp3_true_function(x_2d) + eps
```

### 5.3 `Two_stage/sim_functions/__init__.py` 修改

在 import 段末尾添加：
```python
from .sim_branin_gauss import branin_gauss_simulator, BRANIN_GAUSS_X_BOUNDS
from .sim_branin_student import branin_student_simulator, BRANIN_STUDENT_X_BOUNDS
```

在 `_EXPERIMENT_REGISTRY` 字典中添加：
```python
"branin_gauss":   {"simulator": branin_gauss_simulator,   "bounds": BRANIN_GAUSS_X_BOUNDS,   "d": 2},
"branin_student": {"simulator": branin_student_simulator, "bounds": BRANIN_STUDENT_X_BOUNDS,  "d": 2},
```

---

## 6. 主实验脚本逻辑

### 6.1 `exp_branin/run_branin_compare.py`

**顶部 docstring**：
```
exp_branin: Compare CKME, DCP-DR, hetGP on 2D Branin-Hoo with two noise settings.

  branin_gauss   : Gaussian noise, sigma(x) = 0.4*(4*x1_scaled+1)
  branin_student : Student-t noise, nu(x)=6-4*x1_scaled, sigma(x) same

Usage (from project root):
  python exp_branin/run_branin_compare.py
  python exp_branin/run_branin_compare.py --n_macro 10 --method lhs --n_workers 4
```

**全局常量**：
```python
SIMULATORS = ["branin_gauss", "branin_student"]
MIXED_RATIO = 0.7
R_SCRIPT = _root / "run_benchmarks_one_case.R"

METHOD_COV   = {"CKME": "covered_score",  "DCP-DR": "covered_score_dr",  "hetGP": "covered_interval_hetgp"}
METHOD_WIDTH = {"CKME": "width",          "DCP-DR": "width_dr",          "hetGP": "width_hetgp"}
METHOD_SCORE = {"CKME": "interval_score", "DCP-DR": "interval_score_dr", "hetGP": "interval_score_hetgp"}
```

**`run_one_macrorep(macrorep_id, base_seed, config, simulator_func, out_dir, method, n_grid, params=None)`**：

伪代码逻辑（与 `exp_nongauss/run_nongauss_compare.py` 完全平行）：
```
1. seed = base_seed + macrorep_id * 10000
2. 读取 config 中 n_0, r_0, n_1, r_1, alpha；若 params is None 则用 config["params"]
3. X_cand = get_x_cand(simulator_func, config["n_cand"], random_state=seed+1)
4. stage1 = run_stage1_train(
       n_0=n_0, r_0=r_0,
       simulator_func=simulator_func,
       params=params,
       t_grid_size=n_grid,
       random_state=seed+2,
       verbose=False,
   )
5. stage2 = run_stage2(
       stage1_result=stage1,
       X_cand=X_cand,
       n_1=n_1, r_1=r_1,
       simulator_func=simulator_func,
       method=method,
       alpha=alpha,
       mixed_ratio=MIXED_RATIO,
       random_state=seed+3,
       verbose=False,
   )
6. X_test, Y_test = generate_test_data(...)   # random_state=seed+4
7. eval_result = evaluate_per_point(stage2, X_test, Y_test)
   rows_ckme = eval_result["rows"]
   # 每 row 含 x0, x1 (2D), y, L, U, covered_interval, covered_score, width, interval_score, status
8. case_dir = out_dir / f"macrorep_{macrorep_id}" / f"case_{simulator_func}_{method}"
   rep0_dir = case_dir / "macrorep_0"
   rep0_dir.mkdir(parents=True, exist_ok=True)
   # 保存 X0 (n*r_0, 2), Y0 (n*r_0,), X1 (n_1*r_1, 2), Y1, X_test (n_test, 2), Y_test CSVs
9. bench_csv = case_dir / "benchmarks.csv"
   try:
       bench_df = _run_r_benchmarks(case_dir, bench_csv, alpha, n_grid)
       # 合并 DCP-DR/hetGP 列到 rows_ckme
   except RuntimeError as e:
       print warning; DCP-DR/hetGP 列设为 NaN
10. pd.DataFrame(rows_ckme).to_csv(case_dir / "per_point.csv", index=False)
11. 聚合 mean coverage/width/IS 返回 dict
```

**`main()` CLI 参数**：
```
--config     default: "exp_branin/config.txt"
--output_dir default: None → exp_branin/output/
--n_macro    default: 5
--base_seed  default: 42
--method     choices: [lhs, sampling, mixed], default: lhs
--n_workers  default: 1
```

### 6.2 `exp_branin/config_utils.py`

```python
from Two_stage.config_utils import load_config_from_file, get_config, get_x_cand  # noqa: F401
```

---

## 7. R Benchmark 集成

### 7.1 使用现有的 `run_benchmarks_one_case.R`

**无需修改** `run_benchmarks_one_case.R`。该脚本：
- 读取 `macrorep_0/` 下的 X0.csv, Y0.csv, X1.csv, Y1.csv, X_test.csv, Y_test.csv
- 通过 `d <- ncol(X0)` 自动处理 2D 输入（hetGP 的 `lower/upper` 自动扩展为 length-2 向量）
- 输出 per-point CSV 含 L_dr, U_dr, covered_score_dr, width_dr, interval_score_dr, L_hetgp, U_hetgp, covered_interval_hetgp, width_hetgp, interval_score_hetgp

**调用方式**（与 exp_nongauss 完全相同）：
```python
cmd = ["Rscript", str(R_SCRIPT), str(case_dir), str(bench_csv), str(alpha), str(n_grid)]
subprocess.run(cmd, ...)
```

### 7.2 已知限制

- R 脚本中 Y grid 使用 `seq(min(Y0), max(Y0), ...)` 而非 percentile-based bounds。在 Student-t 极重尾场景（nu→2）下，若 Y0 出现极端值，DCP-DR 的 CDF grid 可能过宽，导致 DCP-DR 结果异常。
- **初版策略**：不修改 R 脚本，记录 hetGP/DCP-DR 失败率；若结果异常再针对性修改。
- hetGP 的 `lower = rep(0.01, d)`, `upper = rep(10, d)` 是 length-scale 范围，对 Branin-Hoo 域（x1∈[-5,10], x2∈[0,15]）可能偏小，初版先运行观察。

---

## 8. 输出结构

```
exp_branin/output/
├── branin_compare_summary.csv          # 汇总表
└── macrorep_{k}/                       # k = 0, 1, ..., n_macro-1
    └── case_{sim}_{method}/            # sim ∈ {branin_gauss, branin_student}
        ├── per_point.csv               # 每测试点一行（见 §8.1）
        ├── benchmarks.csv              # R benchmark 原始输出
        └── macrorep_0/
            ├── X0.csv                  # (n_0*r_0, 2)
            ├── Y0.csv                  # (n_0*r_0,)
            ├── X1.csv                  # (n_1*r_1, 2)
            ├── Y1.csv                  # (n_1*r_1,)
            ├── X_test.csv              # (n_test, 2)
            └── Y_test.csv              # (n_test,)
```

### 8.1 `per_point.csv` 列定义

| 列名 | 类型 | 说明 |
|------|------|------|
| x0 | float | 第一维输入（∈[-5,10]） |
| x1 | float | 第二维输入（∈[0,15]） |
| y | float | 观测值 |
| L | float | CKME 区间下界 |
| U | float | CKME 区间上界 |
| covered_interval | 0/1 | CKME 区间覆盖 |
| covered_score | 0/1 | CKME 分数覆盖 |
| width | float | CKME 区间宽度 |
| interval_score | float | CKME Winkler IS |
| status | str | "in"/"below"/"above" |
| L_dr | float | DCP-DR 下界 |
| U_dr | float | DCP-DR 上界 |
| covered_score_dr | 0/1 | DCP-DR 分数覆盖 |
| width_dr | float | DCP-DR 宽度 |
| interval_score_dr | float | DCP-DR IS |
| L_hetgp | float | hetGP 下界 |
| U_hetgp | float | hetGP 上界 |
| covered_interval_hetgp | 0/1 | hetGP 区间覆盖 |
| width_hetgp | float | hetGP 宽度 |
| interval_score_hetgp | float | hetGP IS |

### 8.2 `branin_compare_summary.csv` 列定义

| 列名 | 说明 |
|------|------|
| simulator | branin_gauss 或 branin_student |
| method | CKME / DCP-DR / hetGP |
| mean_coverage | macrorep 平均 coverage |
| sd_coverage | macrorep 跨宏重复的 SD |
| mean_width | 平均区间宽度 |
| sd_width | SD |
| mean_interval_score | 平均 Winkler IS |
| sd_interval_score | SD |
| n_macroreps | 实际完成的宏重复数 |

---

## 9. Pretrain 脚本规格

### `exp_branin/pretrain_params.py`

结构与 `exp_nongauss/pretrain_params.py` 完全平行：

```
SIMULATORS = ["branin_gauss", "branin_student"]

PARAM_GRID = ParamGrid(
    ell_x_list=[0.5, 1.0, 2.0, 4.0],
    lam_list=[1e-3, 1e-2, 1e-1],
    h_list=[0.1, 0.3, 0.5],
)

pretrain_one(simulator_func, n_pilot=150, r_pilot=5, cv_folds=5, t_grid_size=200, random_state=0)
  → run_stage1_train(... param_grid=PARAM_GRID, cv_folds=cv_folds ...)
  → return result.params.as_dict()

main():
  CLI: --n_pilot (default 150), --r_pilot (default 5), --cv_folds (default 5),
       --t_grid_size (default 200), --seed (default 0), --output (default None),
       --sims (default None = all)
  输出: exp_branin/pretrained_params.json
```

---

## 10. 绘图脚本规格

### `exp_branin/plot_branin.py`

**`--plot_type` 选项**（三种）：

#### 10.1 `--plot_type cov_width`（默认）：Coverage 和 Width vs. x1

**布局**：2行 × 2列

```
         Gaussian          Student-t
Coverage  [ax00]            [ax01]
Width     [ax10]            [ax11]
```

- **x 轴**：测试点的 x1（原始值 -5 到 10），分 `n_bins=25` 个区间，显示 bin 中心
- **3条曲线**：CKME（蓝实线）、DCP-DR（红虚线）、hetGP（绿点划线）；多 macrorep 加 ±1 SD 阴影带
- **Reference line**：Coverage 图加 `y=1-alpha` 水平虚线

**数据读取**：
```
glob(output_dir / "macrorep_*/case_{sim}_{method}/per_point.csv")
```
对每个 macrorep 在 x1 方向分 bin 求均值，再跨 macrorep 取 grand mean ± SD。

#### 10.2 `--plot_type noise`：噪声特征图（oracle）

**布局**：1行 × 2列

```
左：sigma(x1) vs x1（两种噪声共用同一 sigma 函数，只画一条曲线）
右：nu(x1) vs x1（仅 branin_student；标注 nu=2 的危险区域 @ x1=10）
```

纯函数图，无需实验结果，在 x1 ∈ [-5, 10] 上 linspace 画线。

#### 10.3 `--plot_type summary`：汇总 bar chart

**布局**：3行 × 2列

```
             Gaussian          Student-t
Coverage     [ax00]            [ax01]
Width        [ax10]            [ax11]
IS           [ax20]            [ax21]
```

每格：3 methods 的柱状图，误差棒为 SD（n_macro > 1 时）。读取 `branin_compare_summary.csv`。

**CLI 参数（共用）**：
```
--output_dir  default: exp_branin/output
--plot_type   choices: [cov_width, noise, summary], default: cov_width
--n_bins      default: 25  (cov_width only)
--alpha       default: 0.1
--site_method default: lhs
--save        default: None（不保存则 plt.show）
```

---

## 11. Implementation Notes

### 11.1 2D 输入处理

- 所有 simulator 函数的 x 入参需 `np.atleast_2d()` 处理，确保 shape `(n, 2)`。
- `evaluate_per_point` 的 rows 字典通过 `for j in range(d): row[f"x{j}"] = ...` 自动写入 `x0`, `x1` 两列。
- X0/X1/X_test CSV 保存为 `(n, 2)` 矩阵，R 脚本通过 `ncol(X0)` 自动检测 d=2。

### 11.2 nu(x)=2 的处理

`_nu(x)` 用 `np.maximum(2.0, ...)` clip（允许精确等于 2）。在 `chisquare(2.0)` 采样中，数学上合法但方差无穷大，偶尔产生极大值属正常。依赖 percentile-based t_grid 处理。

### 11.3 ell_x 期望值

Branin-Hoo 域大小（x1 跨度 15，x2 跨度 15）远大于 exp_nongauss 的 [0,2π]（跨度约 6.28），CV 后期望 `ell_x` 在 2.0~4.0 范围。CV 搜索网格必须包含这些值。

### 11.4 运行顺序

```bash
# Step 1: 调参（仅需一次）
python exp_branin/pretrain_params.py --n_pilot 150 --r_pilot 5

# Step 2: 主实验
python exp_branin/run_branin_compare.py --n_macro 5 --method lhs

# Step 3: 出图
python exp_branin/plot_branin.py --plot_type cov_width --save exp_branin/output/branin_cov_width.png
python exp_branin/plot_branin.py --plot_type noise    --save exp_branin/output/branin_noise.png
python exp_branin/plot_branin.py --plot_type summary  --save exp_branin/output/branin_summary.png
```

---

## 12. Critical Reference Files

| 文件 | 用途 |
|------|------|
| `Two_stage/sim_functions/exp3.py` | 提供 `exp3_true_function`, `EXP3_X_BOUNDS`；新 simulator 直接 import |
| `Two_stage/sim_functions/sim_nongauss_A1.py` | Student-t simulator 的直接模板 |
| `Two_stage/sim_functions/__init__.py` | 需新增 2 条 import + 2 条 registry 条目 |
| `exp_nongauss/run_nongauss_compare.py` | 主实验脚本的直接模板（几乎逐函数复用） |
| `exp_nongauss/pretrain_params.py` | pretrain 脚本的直接模板 |
| `run_benchmarks_one_case.R` | R benchmark 脚本；无需修改 |
