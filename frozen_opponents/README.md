# Frozen opponents

Checkpoints committed here are played by `agent_code/frozen_agent` when an
experiment lists `frozen_agent` in an opponent list.

They are in git, not referenced out of `runs/`, for the same reason
`initial_model` works that way: `runs/` is ignored and machine-local, and a
self-play arm whose opponent could differ between two machines is not an
experiment. Each is small (about 71 KiB) and each config declares the SHA-256
it expects, so a silently swapped opponent fails the run instead of changing
the result.

| file | route | provenance |
|---|---|---|
| `R02_11_rainbow_seed1005_round10000.npz` | `R02_11` | `runs/m3_rainbow_10000_vs3rb_20260829/jobs/train_seed1005/agent/latest_model.npz`. The current submission model: chosen on validation (7.72), reported on holdout at 7.1270 over 35 repeats (docs/01 section 7.39.6). Committed so it can be played as an opponent by the transfer measurements of section 7.43. |
| `R02_9_seed1001_round05000.npz` | `R02_9` | `runs/m3_lr_decay_5000_vs3rb_20260827/jobs/train_seed1001/agent/latest_model.npz`. The M3 baseline's best training seed, chosen on validation (4.0800) and reported on holdout at 3.8790 over 35 repeats -- the submission candidate until the rainbow arm replaced it (docs/01 section 7.39.6). |
