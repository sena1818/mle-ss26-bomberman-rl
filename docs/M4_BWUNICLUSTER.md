# 在 bwUniCluster 3.0 (KIT) 上跑 M4

> 面向 SLURM 集群。通用手册见 [M4_CLUSTER_RUNBOOK.md](M4_CLUSTER_RUNBOOK.md)。
> 分区与 module 已按你在 `uc3n991` 上的实测确认：`cpu`（3 天上限、192 核/节点）、
> `dev_cpu`（30 分钟）、`devel/python/3.12.3-gnu-14.2`。代码已在 `origin/master`。

## 0. 开跑前的状态

| | 项 | 状态 |
|---|---|---|
| 1 | 代码在 remote 上 | ✅ 已推 `origin/master` |
| 2 | 分区 / module | ✅ 已实测确认（见上） |
| 3 | 集群上的 venv | ⬜ 见 §2，**必须在登录节点建**（计算节点无外网） |

## 1. 参考：查询命令

```bash
sinfo -o "%P %l %c %m"        # 分区名、最大墙钟、每节点核数、内存
module avail python 2>&1 | head -30
ws_allocate m4 60             # 工作区（家目录配额装不下，见 §4）
```

对应改 `scripts/slurm_m4_stage.sh` 顶部的三个值。脚本现在默认 `--partition=cpu`
——**3.0 用 `cpu` / `dev_cpu`，取代了 2.0 的 `single` / `multiple`**，但请用上面那条
`sinfo` 确认，墙钟上限尤其要看。`--time=24:00:00` 和 module 名仍是占位符。

## 1b. ⚠️ Lustre 与小文件

bwUniCluster 3.0 的共享文件系统是 **Lustre**：文件切片摊在多台存储服务器上并行读写，
但**所有 open / close / stat 都要问一台全集群共用的元数据服务器（MDS）**。
大文件顺序 I/O 极快，海量小操作很慢、而且拖累别人。

这个 agent **每个 action 写一条 JSONL**（不是每回合）：一个训练 job 约 **114 万条**，
一个五 seed 的臂约 **570 万条**。

**已修的部分**：句柄现在每个 job 只 open 一次（之前是每条记录 open+write+close），
去掉了约 **1140 万次元数据操作/臂**。记录仍逐条 flush，所以读到的东西没变。

**没修的部分**：写操作**没有批量化**，仍是每个 action 一次 `write + flush`。
所以那个 logger 微基准的「快 84%」**不能外推成整条训练快 84%**——它减轻的是元数据压力，
大量小写入仍可能拖慢吞吐。**如果实测吞吐明显低于 benchmark，这仍是第一嫌疑。**

```bash
df -h "$TMPDIR"      # 节点本地 scratch 还剩多少
```

缓解办法是把 run 目录放在**节点本地 scratch** 上跑，结束时 rsync 回工作区——
bwHPC 官方文档明确说 `$TMPDIR` 在计算节点本地 SSD 上，并且专门建议反复访问数据的
AI 训练用它（[Filesystem Details](https://wiki.bwhpc.de/e/BwUniCluster3.0/Hardware_and_Architecture/Filesystem_Details)）。

**我没有把它自动化**，因为它会改变产物与 provenance 的落盘位置，需要 trap 保证
作业被杀时也能把结果拷回来——那是个要写对的脚本，不是一行 `cd`。
**先在 workspace 上跑 pilot 验证逻辑；只有实测吞吐明显偏低时才值得做这件事。**

## 2. 一次性准备（**登录节点**）

**module 必须和建 venv 时用的是同一个**，否则 venv 里符号链接的解释器和作业里加载的
共享库对不上：

```bash
module load devel/python/3.12.3-gnu-14.2
ws_allocate m4 60
cd "$(ws_find m4)"
git clone git@github.com:sena1818/mle-ss26-bomberman-rl.git bomberman_rl
cd bomberman_rl

python -m venv .venv
.venv/bin/python -m pip install numpy tqdm torch
.venv/bin/python -c "import numpy, torch; print(numpy.__version__, torch.__version__)"

export REPO="$PWD"
export VENV="$PWD/.venv"
export LOG_DIR="$HOME/m4_logs"
```

`.venv` 放仓库根目录，不要放进 job artifact 之下（runtime allowlist 有意排除它）。
**`LOG_DIR` 必须在仓库外**——写进仓库工作区变脏，下一次 `prepare` 直接拒绝。

**先实测吞吐，不要用我这台 Mac 的数字外推**：

```bash
.venv/bin/python scripts/benchmark_cnn.py --rounds 10000 --steps-per-round 300 \
    --output "$LOG_DIR/m4_throughput_$(hostname -s).json"
```

**只有当 `gradient_step_seconds` 与 20.34 ms 同量级时，§5 的时间表才成立。**

### 2b. 先用 `dev_cpu` 验证启动路径（30 分钟上限，专为这个用）

正式排队前，用 20 回合的 smoke 证明 module / venv / REPO / provenance 全链路是通的：

```bash
mkdir -p "$LOG_DIR"
sbatch --partition=dev_cpu --time=00:30:00 --cpus-per-task=8 \
       --export=ALL,REPO="$REPO",VENV="$VENV",LOG_DIR="$LOG_DIR" \
       --output="$LOG_DIR/%x_%j.out" --error="$LOG_DIR/%x_%j.err" \
       --job-name=m4smoke scripts/slurm_m4_stage.sh smoke "$(date +%Y%m%d)"
```

**它不是"几分钟"**：这个 smoke 展开成 **74 个 job**（2 训练 + 72 评估，其中 24 个是
`classic` + 3 个 `rule_based` 的 30 回合对局）。它是一次**完整的集成 smoke**，
不是纯启动检查——`--time=00:20:00` 够不够**没有验证过**。

**所以第一次就用 `dev_cpu` 跑它，正是为了在 30 分钟上限内知道答案。**
如果超时，把 `--time` 提到 30 分钟上限重试；仍不够就说明这个 smoke 对 `dev_cpu` 太大，
改用 `--partition=cpu --time=01:00:00`。

**通了再提 pilot**——`cpu` 队列现在满负荷，排队久，不值得用一次排队去发现 module 名写错了。

## 3. 提交

**必须显式传路径。**`slurm_m4_stage.sh` 默认把 `SLURM_SUBMIT_DIR` 当作仓库（从 clone 里
提交就对了），但显式传更稳，尤其是从别处提交时：

```bash
D=$(date +%Y%m%d)
X="ALL,REPO=$REPO,VENV=$VENV,LOG_DIR=$LOG_DIR"
mkdir -p "$LOG_DIR"
# REQUIRED. SLURM writes these relative to the SUBMIT directory -- this
# repository -- and creates them before the job runs, which makes the checkout
# dirty and makes prepare refuse. There is no safe default, so pass them every
# time.
O="--output=$LOG_DIR/%x_%j.out --error=$LOG_DIR/%x_%j.err"

p=$(sbatch --parsable --export=$X $O --job-name=m4pilot  scripts/slurm_m4_stage.sh pilot  $D)
a=$(sbatch --parsable --export=$X $O --dependency=afterok:$p --job-name=m4anchor scripts/slurm_m4_stage.sh anchor $D)
sbatch --export=$X $O --dependency=afterok:$a --job-name=m4lr1 scripts/slurm_m4_stage.sh lr1e4 $D
sbatch --export=$X $O --dependency=afterok:$a --job-name=m4lr5 scripts/slurm_m4_stage.sh lr5e4 $D
```

`afterok` 就是 gate：`check_pilot.py` 失败让 SLURM job 非零退出，后面的依赖自动取消。
`lr1e4` 与 `lr5e4` 只依赖 anchor、互相独立，队列允许就并行。

**跑完这四个必须停。**步长决策:

```bash
.venv/bin/python scripts/decide_learning_rate.py \
    --anchor runs/m4_anchor_$D --candidate runs/m4_lr1e4_$D --candidate runs/m4_lr5e4_$D --apply
```

然后把选中的值写进四个下游配置的 `agent.learning_rate` **并提交**。
**`slurm_m4_stage.sh` 会拒绝**在决策文件不存在、或四个配置不一致时跑任何增量。

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
