# M2 会话启动 prompt

复制下面整段作为新会话的第一条消息。

---

跑 M2 线。先读 `bomberman_rl/docs/05_路线重构与实验排期.md` 的 §0（含 §0.2、§0.3）和
`docs/01_实验路线_奖励_课程训练设计.md` 的 §7.5.7，那是当前 canonical 基线；§7.5.1–§7.5.6 是
历史记录，别拿它们的数字做比较。

## 当前状态

- HEAD 应为 `e5e6d5f` 或更新。canonical R01 基线 = commit `5c430c8` 的无滞后 runtime，
  四臂结果在 `runs/r01_*_nolag_20260826/`（本机只存了轻量产物；完整 run 目录在 Hetzner）。
- M1 已收尾，不要再改 R01 的 runtime 或 reward 表。
- M2 的 5 个配置已存在，不需要新建：

  | 配置 | 相对上一级的唯一改动 | 对应实验 |
  |---|---|---|
  | `experiments/m2_r01_a03_e01_t02.json` | 训练任务 classic → loot-crate | 实验 1（先跑，它是后面所有臂的对照） |
  | `experiments/m2_r01_a06_e01_t02.json` | 加 A06 势函数塑形 | 实验 2 |
  | `experiments/m2_r01_a06_e01_t02_n3.json` | n_step 1 → 3 | 实验 3 |
  | `experiments/m2_r01_a06_e01_t02_n5.json` | n_step 1 → 5 | 实验 3 |
  | `experiments/m2_r01_a06_e01_t02_replay.json` | 加 replay + target network | 实验 4 |

  每个都是 5 seed × 500 局 × 365 job，主评估 loot-crate，外加 `classic_transfer` 与
  `coin_heaven_diagnostic` 两个 suite。

## 顺序

严格按上表顺序，**一次只加一样东西**。实验 1 必须先跑完——它是实验 2/3/4 的对照臂。
实验 1 的对照是 `runs/r01_a03_e01_nolag_20260826`（canonical A03_E01，classic 上训练），
唯一差异是训练任务，所以 `compare_runs.py` 需要 `--allow-multiple-differences`，
而且**主评估分数跨任务不可直接比**（docs/01 §2.2）——跨任务的比较走 `classic_transfer` suite
和行为指标（`crates_per_round` / `approximate_safe_bomb_rate` / `wait_fraction` / `bomb_rate`）。

实验 1 的预注册预期写在配置的 `_predeclared_design_numbers` 里：`crates_per_round` 相对 T03 上升，
5/5 个 seed 都要有放弹样本。跑完先对这两条作判定。

## 怎么跑

Hetzner，8 核：

- 地址 `root@46.224.43.9`，密钥 `~/.ssh/automobility_hetzner`
- 仓库 `/root/bomberman-r01`，解释器 `.venv/bin/python`（numpy 2.5.2 + tqdm 4.70.0，
  **没有 pygame 也没有 torch**——`--no-gui` 不需要 pygame，torch 是惰性导入，M1–M3 用不到）
- 单臂 `run --jobs 8` 约 5–6 分钟

```bash
ssh -i ~/.ssh/automobility_hetzner root@46.224.43.9 \
  'cd /root/bomberman-r01 && git fetch -q origin && git reset --hard -q origin/master && \
   nohup .venv/bin/python scripts/run_experiment.py run \
     --config experiments/m2_r01_a03_e01_t02.json \
     --run-id m2_t02_20260827 --jobs 8 > /root/m2_t02.log 2>&1 &'
```

日志写 `/root/` 而不是仓库内，否则工作区变脏、下一次 `prepare` 会被拒。

## 六条硬规矩

1. **`prepare` 要求干净的已提交 checkout。**每个 job 执行前还会再验一次 provenance
   （`experiment_lib.py`），工作区一旦有未提交改动就直接拒跑。长实验绝不要用 `--allow-dirty`。
2. **本机可能有别的会话在改同一个仓库。**要在本机跑长实验就用 `git worktree add --detach <path> <commit>`
   开一个独立干净树，别去动别人的工作区。
3. **凡经过选种且 `n = 3` 的数字，必须同时报告全部 training seed 的分布。**
   这条是血的教训：coin-heaven 的 `34.08 ± 3.57` 实际是五个 seed `{9.36, 14.78, 1.41, 34.08, 1.92}`
   里的离群点，害得一个 gate 被误判为「全数通过」。
4. **别从旧 checkpoint 续训。**每个臂都 fresh start，否则会把不可比的训练历史带进来。
5. **写文档时保留原表、并列新表**，在旧节顶部加指向新节的提示。不要覆盖已发表的数字——
   翻车记录本身有价值。
6. **跑完压缩。**`scripts/prune_runs.py --run-dir runs/<id> --compress-logs --apply`，
   单臂能回收约 2 GiB（聚合器透明读 `.jsonl.gz`，压完仍可重新汇总）。

## 已知的坑

- `agent.jsonl` 每步一条记录，一个 500 局训练 job 约 7 MB；`framework_game.log` 更大。
  单臂未压缩约 2.4 GB。
- 运行时会把「本步之后回合是否结束」的预测与框架实际行为逐条比对，分歧计入 `round_end` 的
  `round_end_mispredictions`。**每次跑完都检查它是否为 0**；非 0 说明烟雾阶段近似被触发，
  要先查清楚再报结果。
- 跨平台（macOS ARM / Linux）、跨并发度（串行 / `--jobs 8`）结果逐位一致，已验证两次。
  所以出现差异时不要归因于环境，一定是代码或配置变了。

## 汇报要求

每个臂跑完给：完整性（`exit0=n/365`、`failed_attempts`、provenance 的 commit 与 dirty）、
holdout @500 对照表、逐 seed 分布、`round_end_mispredictions`、预注册预期的判定。
然后按上面第 5 条写进 docs/01 和 docs/05。
