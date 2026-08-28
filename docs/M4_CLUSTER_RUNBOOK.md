# M4 集群执行手册

> **在 bwUniCluster 3.0 上跑请看 [M4_BWUNICLUSTER.md](M4_BWUNICLUSTER.md)**，那份是当前的。
> 本文是 scheduler-agnostic 的通用说明；§2 的臂顺序与 §0 的 benchmark 路径**曾经是旧的，已同步**。

> 面向：把 M4 的臂送上计算集群的人。本机不需要跑任何东西。
> 配方与判据见 [06 文档](06_M4学习式空间表示线设计.md)；集群通用流程见 [cluster_execution.md](cluster_execution.md)。

## 0. 一次性准备

M4 是**唯一**需要 PyTorch 的线（M1–M3 纯 NumPy）。目标机器上：

```bash
python -m pip install torch          # Hetzner 的 .venv 目前没有
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**先实测，不要外推**（这是 §5.4 的规矩，本项目已经因为外推错过一次 2.1×）：

```bash
python scripts/benchmark_cnn.py --rounds 10000 --steps-per-round 300 \
    --output "$LOG_DIR/m4_throughput_<host>_<date>.json"     # 仓库外！
```

它会打印每个梯度步的完整耗时、env steps/s 和推理余量。**只有当这台机器的
`gradient_step_seconds` 与 Apple M4 的 20.34 ms 同量级时，§8 的排期才适用。**

有 GPU 的话，同一条命令加环境变量再测一次，两个数字都记进报告：

```bash
BOMBERMAN_TORCH_DEVICE=cuda python scripts/benchmark_cnn.py --rounds 10000 --steps-per-round 300
```

**默认是 CPU，且提交的 agent 必须能在 CPU 上跑**——checkpoint 存的是 numpy 数组，
与设备无关，所以 GPU 上训练、CPU 上提交是安全的。

## 1. 资源

| | 每个训练 job |
|---|---|
| 进程数 | 1（`BOMBERMAN_TORCH_THREADS` 默认 1，故意的：并行是跨进程的） |
| 内存 | **约 0.5 GB**（100k replay 以 uint8 存储 = 406 MB + 模型与框架） |
| 磁盘 | 5 seed 一臂压缩前约 **2–3 GB**，`prune_runs.py` 后约 1/10 |
| 时长 | 见 §0 实测；Apple M4 单线程约 **4 h/seed** |

一臂 = **5 个训练 job + 330 个评估 job = 335 job**。评估 job 不分配 replay，内存很小。

**并行**：`--jobs N` 是进程级的，阶段内并发、阶段间保序。N 取核数。
**不需要任何并行化改造**；`prepare` + per-job JSON 已经是 scheduler-agnostic 的。

## 1b. 一条命令跑完整条线

```bash
./scripts/run_m4_line.sh --jobs 8 --date $(date +%Y%m%d)
```

它按顺序做：装机检查 → **吞吐实测** → pilot → gate → anchor → gate → 两个步长臂
→ **在步长决策处停下并退出**。每个臂跑完自动 `aggregate` + `check_pilot` + `prune`。

- **它不会一口气跑完整条线。**步长决策之后必须人工执行
  `scripts/decide_learning_rate.py --apply`、改四个下游配置并提交，
  再用 `--lr-settled` 重跑才会继续。
- `--lr-settled` 会调用 `decide_learning_rate.py --verify`，
  **要求决策文件存在、且四个下游臂解析出的学习率都等于决策值**。
- **gate 不过就停**，不会继续烧后面的臂。
- 已完成的臂（存在 `evaluation_summary.json`）自动跳过，**中断后重跑安全**。
- `--from anchor` 从指定阶段开始；`--dry-run` 只打印计划。
- 日志写 `~/m4_logs/`（**仓库外**，否则工作区变脏、下一次 `prepare` 被拒）。

后台提交：

```bash
nohup setsid ./scripts/run_m4_line.sh --jobs 8 --date $(date +%Y%m%d) \
    > ~/m4_line.log 2>&1 < /dev/null & disown
```

下面 §2–§5 是这条命令内部做的事，手工分步执行时看它们。

## 2. 顺序（每一步的产物决定下一步跑不跑）

```text
P   pilot        5000 局 ×2 seed    38 job    ~1 h    ← 必须先跑完并看 §3
1   anchor      10000 局 ×5 seed   335 job    ~4.4 h/seed   gate: G-A
2a  步长 1e-4   10000 局 ×5 seed   335 job    ┐ 互相独立，可并行
2b  步长 5e-4   10000 局 ×5 seed   335 job    ┘
--- 强制停一次：scripts/decide_learning_rate.py --apply ---
3   对手暴露    10000 局 ×5 seed   335 job
4   A03 去塑形  10000 局 ×5 seed   335 job
5   BC 热启动   10000 局 ×5 seed   335 job
6   dueling     10000 局 ×5 seed   335 job
```

**步长决策是硬 gate**：`slurm_m4_stage.sh` 与 `run_m4_line.sh` 都会拒绝在
`runs/m4_anchor_<date>/learning_rate_decision.json` 不存在时跑任何增量。

## 3. pilot 的验收条件（**不过就不要开正式臂**）

```bash
python scripts/run_experiment.py run \
  --config experiments/m4_r07_a06_e09_t02_pilot.json \
  --run-id m4_pilot_<date> --jobs 4
python scripts/check_pilot.py --run-dir runs/m4_pilot_<date>
```

`check_pilot.py` 逐条打印六项检查并以退出码表示通过与否。
前四项本机已经验过（`diagnostics/m4_pilot_e09_20260827.json`），
**第五项「validation 分数是否上升」还没验过**——评估腿没跑。

## 4. 正式臂

```bash
# 必须是干净的已提交 checkout；长实验绝不用 --allow-dirty
python scripts/run_experiment.py run \
  --config experiments/m4_r07_a06_e09_t02_anchor.json \
  --run-id m4_anchor_<date> --jobs <cores>
```

日志写仓库外（否则工作区变脏，下一次 `prepare` 被拒）：

```bash
nohup setsid python scripts/run_experiment.py run --config <...> --run-id <...> --jobs 8 \
   > /root/<run-id>.log 2>&1 < /dev/null & disown
```

等待时用 `pgrep -f "^\.venv/bin/python scripts/run_experiment"`——**必须带 `^` 锚点**。

## 5. 跑完

```bash
python scripts/aggregate_results.py --run-dir runs/<id>            # 不加 --promote
python scripts/check_pilot.py --run-dir runs/<id> --training-only  # 健康检查同样适用于正式臂
python scripts/prune_runs.py --run-dir runs/<id> --drop-runtime --compress-logs --apply
python scripts/verify_run_archives.py
```

**引用 `evaluation_summary.json` 的 `reportable_result`，不要引用 `selected_checkpoint`**——
后者是在 `train_seed × checkpoint_round` 上取的最大值，系统性偏高。
带对手的臂必须并列报告 `coins_share`。

## 6. 三个容易踩的坑

1. **`rule_based_agent` 不确定**（`callbacks.py:69` 调用无参 `np.random.seed()`）。
   带对手的臂**训练和评估都不可复现**，每个数字都带这层噪声。
2. **`round_end_mispredictions` 读终值**，不要逐回合求和（它是累计计数器）。
3. **`--slim-copy` 的产物在 `DEST/<run>/<run>/`**，rsync 到已存在目录会嵌套不会合并。
