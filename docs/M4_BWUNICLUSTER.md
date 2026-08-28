# 在 BwUniCluster 上跑 M4

> 面向 SLURM 集群。通用手册见 [M4_CLUSTER_RUNBOOK.md](M4_CLUSTER_RUNBOOK.md)。
> **本文里标 ⚠️ 的三个值我无法替你确认**，第一次提交前必须自己查。

## 0. 现在还不能直接提交，有三件事挡着

| | 阻塞项 | 怎么解 |
|---|---|---|
| 1 | **M4 的代码还没推到 remote。**`prepare` 要求干净的**已提交** checkout，集群上是 clone 出来的，所以必须先推 | 决定是否把 `m4-prep` 合进 `master` 并 `git push` |
| 2 | **集群上没有 torch，而计算节点通常没有外网** | 在**登录节点**建好 venv（见 §2） |
| 3 | **三个站点相关的值我不能替你确认** | `sinfo`、`module avail`，见 §1 |

## 1. ⚠️ 三个必须自己确认的值

```bash
sinfo -o "%P %l %c %m"        # 分区名、最大墙钟、每节点核数、内存
module avail python 2>&1 | head -30
ws_allocate m4 60             # 工作区（家目录配额装不下，见 §4）
```

对应改 `scripts/slurm_m4_stage.sh` 顶部的 `--partition`、`--time`、`module load`。
脚本里写的 `single` / `24:00:00` / `devel/python/3.11` **是占位符，不是建议值**。

## 2. 一次性准备（**登录节点**）

```bash
ws_allocate m4 60                       # 或你们站点的等价命令
cd "$(ws_find m4)"
git clone <your-remote> bomberman_rl && cd bomberman_rl
git checkout <the commit you pushed>

python -m venv .venv
.venv/bin/python -m pip install numpy tqdm torch
.venv/bin/python -c "import numpy, torch; print(numpy.__version__, torch.__version__)"

export REPO="$PWD" VENV="$PWD/.venv" LOG_DIR="$HOME/m4_logs"
```

**`.venv` 放在仓库根目录**，不要放进 job artifact 之下（runtime allowlist 有意排除它）。
`LOG_DIR` **必须在仓库外**——写进仓库会让工作区变脏，下一次 `prepare` 直接拒绝。

**先实测吞吐，不要用我这台 Mac 的数字外推**：

```bash
.venv/bin/python scripts/benchmark_cnn.py --rounds 10000 --steps-per-round 300 \
    --output "$LOG_DIR/m4_throughput_$(hostname -s).json"
```

**只有当 `gradient_step_seconds` 与 20.34 ms 同量级时，§5 的时间表才成立。**
集群单核通常比 Apple M4 慢 1.5–2 倍，慢多少直接乘到墙钟上。

## 3. 提交（每个 stage 一个 SLURM job，用依赖串起来）

```bash
D=$(date +%Y%m%d)
p=$(sbatch --parsable --job-name=m4pilot  scripts/slurm_m4_stage.sh pilot  $D)
a=$(sbatch --parsable --dependency=afterok:$p --job-name=m4anchor scripts/slurm_m4_stage.sh anchor $D)
sbatch --dependency=afterok:$a --job-name=m4lr1 scripts/slurm_m4_stage.sh lr1e4 $D
sbatch --dependency=afterok:$a --job-name=m4lr5 scripts/slurm_m4_stage.sh lr5e4 $D
```

`afterok` 意味着**gate 不过就不往下跑**：`check_pilot.py` 失败会让 SLURM job 以非零退出，
后面的依赖自动取消。这正是我们要的行为。

`lr1e4` 与 `lr5e4` 只依赖 anchor、互相独立，**队列允许就并行**。

**跑完这四个就停。**步长决策见 [06 文档 §8](06_M4学习式空间表示线设计.md)。

## 4. 三个容易踩的坑（都不是 SLURM 常识，是这个工作负载特有的）

**① 这个负载不跨节点。**`run --jobs N` 是在**一台**机器上开 N 个子进程。
`--nodes=2` 只会让第二个节点闲着。要用更多机器，**并行提交独立的 stage**，
不是把一个 stage 摊到多节点。

**② 线程超订会让它比笔记本还慢。**N 个并发 job 每个都默认"用满所有核"时节点会颠簸。
脚本里这四行是必需的，不是整洁：

```bash
export BOMBERMAN_TORCH_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

**③ 家目录配额装不下。**M3 的经验：一个 2000 局的臂压缩前约 4 GiB，5000 局约 4.1 GiB
（压缩后 288 MiB）。M4 一个 10,000 局 × 5 seed 的臂**压缩前估计 8–10 GB**，
第一批四个 stage 合计 **约 25–30 GB**。用工作区，别用 `$HOME`。
脚本每个 stage 跑完会自动 `prune_runs.py --compress-logs`。

## 5. 时间估计

以 Apple M4 单核实测的 **20.34 ms/梯度步 → 191 env steps/s** 为基准，
一个 10,000 局的臂约 **4.4 h/seed**；5 个 seed 并行需 5 核。

| stage | job 数 | 5 核并行 | 16 核并行 |
|---|---:|---:|---:|
| pilot | 38 | ~1 h | ~0.5 h |
| anchor | 335 | ~4.7 h | ~4.7 h（训练受 seed 数限制，评估变快） |
| lr1e4 | 335 | ~4.7 h | ~4.7 h |
| lr5e4 | 335 | ~4.7 h | ~4.7 h |

- **训练腿只有 5 个 job**，所以超过 5 核不会让训练更快；多出来的核加速的是 330 个评估 job。
- **`lr1e4` 与 `lr5e4` 并行** → 第一批墙钟 ≈ `1 + 4.7 + 4.7` ≈ **10.5 小时**（Mac 基准）。
- **集群单核若慢 1.7 倍** → **约 18 小时**。请按 §2 的实测数字自己乘。
- 第二批（对手 / A03 / BC / dueling）四个 stage 同样各 ~4.7 h，互相独立可全并行。

**单个 stage 最长约 5 小时**（Mac 基准）或 **8–9 小时**（保守），所以
`--time=24:00:00` 有充足余量；但**如果你们分区上限是 8 小时**，anchor 可能不够，
那就要把 `--cpus-per-task` 提到 ≥5 保证 5 个训练 seed 真并行，并重新估。
