# M3 手工特征线：会话启动 prompt

复制下面 `---` 之后的整段作为新会话的第一条消息。

---

把 M3 手工特征线推到最好。表示固定为 `handcrafted_v3`，**不要换 CNN、不要改特征**，
可以动的是训练预算、算法、网络容量与训练配方。

## 先读这些（按顺序，不要跳）

1. **`docs/05_路线重构与实验排期.md` 的 §0 全部**（§0.1 → §0.13）。这是项目当前状态的唯一入口，
   按时间顺序写，后面的节会更正前面的节。
2. **`docs/01_实验路线_奖励_课程训练设计.md` 的 M3 线**，阅读顺序是
   **§7.12 → §7.10 → §7.11 → §7.13 → §7.14 + §7.15 → §7.16**
   （§7.12 是补记，写作时间晚于 §7.10，所以节号和时间顺序不一致）。
3. **`docs/01` 的 §2.1、§2.2**：为什么在 `T02` 训练而在 `T03` 评估，以及**不同任务的分数不可直接比**。
4. **`docs/01` 的 §8.1–§8.3**：指标定义、行为 gate、G1/G2/G3 的阈值与含义。

**§7.5.1–§7.5.6 与 §7.14 的部分结论是历史记录或已被更正，不要拿它们的数字做比较。**

## 当前最好的臂

| | 配置 | 结果 |
|---|---|---|
| **`M3.3L`** | `m3_3l_r02_3_a06_e01_t02_n5_v3_2000.json` | loot-crate 32.622 / 50，classic solo 5.433 / 9，**G2 6/6** |
| `M3.3LX` | 同上 + 对手评估 suite | classic + 3 对手 **2.087** |
| `M3.3XL` | `m3_3xl_r02_3_a06_e01_t02_n5_v3_5000.json` | 5000 局，**2026-08-27 启动，结果待收** |

路线 `R02_3` = `handcrafted_v3`(62 维) + MLP 62-64-32-6 + Q-learning
+ Adam 1e-3 + Huber + grad clip 10 + replay 50k/32/min1000 + target 每 500 步
+ n=5 + A06 势函数 + E01 + 在 `T02` solo 上训练。

## 已经被证据关掉的方向（不要重做）

- **加更强的手工逃生特征**：§7.16.3 实测天花板是自炸死亡的 **11.9%**。`v3` 已经把主要矛盾解决了
  （「致命岔路不可分辨」94.4% → 34.7%）。
- **加大死亡惩罚**：反复论证过会压低放弹率，且 §7.16.5 证明奖励与塑形都已指对方向。
- **escape 专项地图**：`SCENARIOS` 只有 `{CRATE_DENSITY, COIN_COUNT}` 两个旋钮
  （`settings.py:10`），自定义布局必须改官方 `environment.py::build_arena()`，与提交口径冲突。
- **C02 课程**：§7.13.5 显示一个从未见过 `T01` 的 agent 已经把 `T01` 打到 49.25/50，
  课程第一级没有东西可教。
- **直接在 `classic` 上训练**：§7.14.3 实测输给 `T02` 训练 + 迁移（1.389 vs 2.227）。
- **把对手训练当默认**：§7.15.3，score 上测不出收益，但**它确实把存活率从 3.9% 提到 17.4%**，
  代价是单人 classic 掉 68%。视目标而定，不是无脑选项。

## 当前诊断出的瓶颈（这是主线）

§7.16.5：**剩下 38.2% 的自炸死亡是值函数问题，不是表示问题。**
那些死亡的最后一拍，127 次存在更安全的移动，其中 **117 次（92.1%）救命方向在 `handcrafted_v3`
里清清楚楚可见，而 agent 选了 WAIT（85 次）**。A06 塑形也没有鼓励 WAIT（平均塑形差 −0.1290）。

**信息给了、奖励也指对了，Q 值仍然把 WAIT 排在前面。**

## 建议的实验顺序（每次只改一样）

| # | 改动 | 依据 | 备注 |
|---|---|---|---|
| 1 | **预算 2000 → 5000** | §7.13.2 曲线到 2000 局仍未饱和，最后一段最陡 | 已启动，`m3_3xl_v3_5000_20260827` |
| 2 | **`algorithm: q_learning → double_dqn`** | 值函数高估是 WAIT 被排前的经典成因；`double_dqn` 已在 `DECLARATIVE_ROUTE_VALUES` 里 | 需新建 route（照 `R02_3` 抄，只改 algorithm） |
| 3 | **网络容量 (64,32) → (128,64)** | `mean_hidden_zero_fraction ≈ 0.62`，六成 ReLU 不激活 | 单因素 |
| 4 | n-step 重扫（3 / 5 / 8） | M2 在线性模型上测过 n=5 最好，MLP 上没重测 | 优先级低于 1–3 |

**第 2、3 项需要在 `agent_code/research_agent/config.py` 与 `scripts/experiment_lib.py`
的 `IMPLEMENTED_ROUTES` 两处都注册新 route**（两个注册表是独立的，只加一处会在 `prepare` 时报
`Route 'X' is not implemented`）。

## 判定标准：不要只看 score

每跑完一个臂，**除了 G2 六项，还要重跑诊断**：

```bash
python scripts/diagnose_bomb_escape.py --run-dir runs/<id> --job-prefix eval_round0<final>
```

看 §7.16.5 那组数字有没有动：**「最后一拍看得见出口却选了 WAIT」的次数应当下降。**
如果 score 涨了而这个数不动，说明多出来的预算买到的是采集而不是生存——那是另一回事，要分开说。

对手评估用 `scripts/attribute_deaths.py`（解析框架日志，不重放，因此不受对手不确定性影响）。

## 怎么跑

Hetzner，8 核：地址 `root@46.224.43.9`，密钥 `~/.ssh/automobility_hetzner`，
仓库 `/root/bomberman-r01`，解释器 `.venv/bin/python`（numpy + tqdm，**没有 pygame 也没有 torch**）。

```bash
ssh -i ~/.ssh/automobility_hetzner root@46.224.43.9 \
  'cd /root/bomberman-r01 && nohup setsid .venv/bin/python scripts/run_experiment.py run \
     --config experiments/<config>.json --run-id <id> --jobs 8 \
     > /root/<id>.log 2>&1 < /dev/null & disown' > /dev/null 2>&1
```

**日志写 `/root/` 而不是仓库内**，否则工作区变脏、下一次 `prepare` 会被拒。

等待时用 `pgrep -f "^\.venv/bin/python scripts/run_experiment"` —— **必须带 `^` 锚点**，
进程 cmdline 是相对路径 `.venv/bin/python`，用绝对路径匹配会一直匹配不上，
会让你误以为跑完了（这个坑踩过两次）。

成本参考（665 job）：2000 局约 5.5 分训练 + 13.5 分评估；**评估比训练贵**，
因为训练是 5 job × N 局，评估是 660 job × 30 局。

## 八条硬规矩

1. **`prepare` 要求干净的已提交 checkout**，每个 job 执行前还会再验一次 provenance。
   长实验绝不用 `--allow-dirty`。
2. **本机可能有别的会话在改同一个仓库。**本机跑长实验用 `git worktree add --detach`。
3. **凡经过选种的数字，必须同时报告全部 training seed 的分布。**
4. **别从旧 checkpoint 续训**，每个臂 fresh start。
5. **写文档保留原表、并列新表**，在旧节顶部加指向新节的提示。**不要覆盖已发表的数字。**
6. **跑完压缩**：`scripts/prune_runs.py --run-dir runs/<id> --drop-runtime --compress-logs --apply`，
   2000 局单臂能回收约 4 GiB。再用 `--slim-copy` 拉一份轻量副本回本机。
7. **每次跑完检查 `round_end_mispredictions` 是否为 0**（`scripts/` 里的 integ 检查）。
8. **每个分数都要写明「哪个 scenario + 有没有对手」**，跨这两个维度的数字一律不并列
   （§7.15.2 有一张因此作废的表）。

## 两个必须知道的环境事实

- **`rule_based_agent` 不确定**：`callbacks.py:69` 调用 `np.random.seed()`（无参数）。
  **带对手的臂不可复现**——同一个 eval job 连跑四次，30 局合计 score 是 76 / 64 / 69 / 62。
  §0.2 的「逐位一致」只适用于 solo 臂。
- **solo 臂是确定的**，已由 `M3.3LX` 复现 `M3.3L` 逐位验证（§7.14.1）。

## 一条方法论提醒

本轮有**两个**从代码推理得出、听起来完全合理、却被实测推翻的机制假设：
§7.10.5 的「势函数偏向 WAIT」与 §7.16.4 的「对手堵死逃生路」。
**共同点是先有故事、后有数据。**

有效的做法一直是反过来：`v2`/`v3` 之所以成功，正是因为**先量出了 94.4% 这个具体数字才动手**。
提假设可以，但**在数据之前不要写进结论，也不要据此设计实验**。
