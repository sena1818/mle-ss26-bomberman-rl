# M4 集群执行：会话启动 prompt

复制下面 `---` 之后的整段作为新会话的第一条消息。

---

在 bwUniCluster 3.0 (KIT) 上把 M4 线跑完。**你的职责只有执行、读结果、判断 gate**，
不要改实验设计——设计已经冻结在 `docs/06_M4学习式空间表示线设计.md`。

## 先读这些（按顺序）

1. **`docs/M4_BWUNICLUSTER.md`** —— 集群手册，命令都在里面
2. **`docs/06_M4学习式空间表示线设计.md` 的 §0b、§1、§6、§8** —— 修订记录、判据、配方、臂顺序
3. `docs/M4_CLUSTER_RUNBOOK.md` —— scheduler-agnostic 的通用说明

## 环境（登录节点已就绪）

```bash
cd /pfs/work9/workspace/scratch/hd_cn362-m4/bomberman_rl
export REPO="$PWD" VENV="$PWD/.venv" LOG_DIR="$HOME/m4_logs"
export X="ALL,REPO=$REPO,VENV=$VENV,LOG_DIR=$LOG_DIR"
export O="--output=$LOG_DIR/%x_%j.out --error=$LOG_DIR/%x_%j.err"
D=$(date +%Y%m%d)
```

venv 已建（Python 3.12.3 + numpy 2.5.2 + torch 2.13.0）。分区 `cpu`（3 天上限、192 核/节点）、
`dev_cpu`（30 分钟）。module 是 `devel/python/3.12.3-gnu-14.2`。

## 已经完成的

| | 结果 |
|---|---|
| **smoke**（74 job，`dev_cpu`） | ✅ `COMPLETED`，`10 passed, 0 failed`，6 分 53 秒。启动链路已证明 |
| **benchmark**（计算节点实测） | `gradient_step` **14.78 ms**、**246 env/s**、**3.4 h/seed**（比 Apple M4 快 1.38×）；torch 冷 import 9 秒（Lustre，可忍） |
| **日志体积** | `framework_game.log` 60%、`agent.jsonl` 37%；eval job 占 97%。`/pfs/work9` 剩 1.7 PB，**空间不是约束，不要优化它** |

## 现在的状态

`m4pilot`（job 6733100）在 `cpu` 队列排队中。一个 SLURM job = 一个 stage，
stage 内部用 `--jobs 8` 并发跑 38 个实验 job（2 训练 seed + 36 评估）。

## 第一批要跑的四个 stage

```text
pilot     38 job   ~2 h    gate: check_pilot 六项
anchor   335 job   ~3.4 h  gate: G-A
lr1e4    335 job   ~3.4 h  ┐ 只依赖 anchor，互相独立，可并行
lr5e4    335 job   ~3.4 h  ┘
--- 硬 gate：步长决策，见下 ---
```

```bash
p=$(sbatch --parsable --export=$X $O --partition=cpu --time=06:00:00 --cpus-per-task=8 --mem=16G --job-name=m4pilot  scripts/slurm_m4_stage.sh pilot  $D)
a=$(sbatch --parsable --export=$X $O --partition=cpu --time=08:00:00 --cpus-per-task=8 --mem=16G --dependency=afterok:$p --job-name=m4anchor scripts/slurm_m4_stage.sh anchor $D)
sbatch --export=$X $O --partition=cpu --time=08:00:00 --cpus-per-task=8 --mem=16G --dependency=afterok:$a --job-name=m4lr1 scripts/slurm_m4_stage.sh lr1e4 $D
sbatch --export=$X $O --partition=cpu --time=08:00:00 --cpus-per-task=8 --mem=16G --dependency=afterok:$a --job-name=m4lr5 scripts/slurm_m4_stage.sh lr5e4 $D
```

`afterok` 就是 gate：`check_pilot.py` 失败让 SLURM job 非零退出，后续依赖自动取消。

## pilot 回来后必须看的三件事

1. **`check_pilot` 的检查 5（validation 分数是否上升）——这是整条线里唯一还没验过的操作性问题。**
   它决定后面三个臂值不值得跑。
2. **真实磁盘**：`du -sh runs/m4_pilot_$D`（压缩前后），用它替换文档里的外推估计。
3. **实际吞吐** vs benchmark 的 246 env/s。明显更低就先怀疑 Lustre，`df -h "$TMPDIR"`。

## 步长决策：硬 gate，绕不过去

四个增量臂（opponents / no_shaping / bc / dueling）**会被拒绝**，直到：

```bash
$VENV/bin/python scripts/decide_learning_rate.py \
    --anchor runs/m4_anchor_$D --candidate runs/m4_lr1e4_$D --candidate runs/m4_lr5e4_$D --apply
```

预注册规则：候选臂只有在**末段池化 validation 分数超过 anchor 且差值 > 4.7** 时才算赢，
**否则保留 anchor 的 2.5e-4**。然后把选中的值写进四个下游配置的 `agent.learning_rate` **并提交**。
`--verify` 会检查决策值等于四个臂**各自 resolved** 的学习率。

**为什么要停**：`docs/05` §0.20 发表过「加容量有害」的结论，后来发现是**步长没调**造成的伪影。
在没调好的基座上测增量，增量的符号会反过来。

## 六条硬规矩

1. **`prepare` 要求干净的已提交 checkout。**长实验绝不用 `--allow-dirty`。
2. **`--output` / `--error` 必须传到 `$LOG_DIR`。**SLURM 默认写到提交目录（=仓库），
   而且在作业启动瞬间就创建文件，会让 checkout 变脏、`prepare` 拒绝。**这个坑踩过一次。**
3. **`$LOG_DIR` 必须在仓库外。**benchmark 的输出同理。
4. **不跨节点。**`--jobs N` 是单节点内的多进程；`--nodes=2` 只会让第二个节点闲着。
   要更多机器就并行提交独立的 stage。
5. **线程限制是必需的**（脚本已设）：`BOMBERMAN_TORCH_THREADS` / `OMP_NUM_THREADS` /
   `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` 全为 1。
6. **引用 `evaluation_summary.json` 的 `reportable_result`，不要引用 `selected_checkpoint`**
   ——后者在 `train_seed × checkpoint_round` 上取最大值，系统性偏高。
   带对手的臂必须并列报告 `coins_share`。

## 两件与集群无关但会咬人的事

- **Hetzner 上有另一条线（M3）在跑。**推 origin 不影响它，**但那边在 run 结束前不要 `git pull`**
  ——`verify_job_provenance` 每个 job 都比对 commit，HEAD 一动后续 job 全部失败。
- **`rule_based_agent` 不确定**（`callbacks.py:69` 调用无参 `np.random.seed()`）。
  带对手的臂**训练和评估都不可复现**；判定必须用重复测量（≥5 次重跑末 checkpoint 的对手评估），
  不能拿单次差值对 0.246 门槛下结论。

## 第二批（步长定了之后，不是现在）

`opponents` → `no_shaping` → `bc` → `dueling`，各 335 job / ~3.4 h。
**对手臂是重点**：它是唯一能回应 §8.0b 那个 73.2% 缺口的臂，也是唯一能让
`opponents` 平面和 `living_opponents` 拿到非零梯度的臂。

`board_egocentric_v3`（对手 `bombs_left`）**已预注册为下一阶段候选**，
触发条件是 v2 对手臂仍卡在那个缺口。**不要提前把它塞进对手臂**——
那样「多人训练分布」和「可放弹标记」同时变，成败都归因不了。
