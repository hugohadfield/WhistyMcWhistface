# Parallel tournament execution — measured 1, 2, and 4 game workers

**Date:** 31 October 2026 Pacific / 1 November 2026 UTC  
**Implementation:** [PR #18](https://github.com/hugohadfield/WhistyMcWhistface/pull/18)  
**Experiment:** [GitHub Actions run #41284788855](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41284788855)  
**Runner:** GitHub-hosted Ubuntu, Python 3.11; `os.cpu_count() == 4`.  
**Workload:** The existing 13-round three-player head-to-head trump tournament, with two Monte Carlo bots at **1,024 rollouts per non-forced card decision** and one random opponent. 12 matches in six paired deals using base seed 61234.

## Design

The `benchmarks/trump_tournament.py` CLI now supports `--workers N`. The new process pool executes **independent whole games**, not individual rollout branches. Each child runs its own Python interpreter and global RNG, which the game seeds explicitly from its game/pair index. Within each AI decision, the existing evaluator continues to use **one** Monte Carlo rollout worker.

- The original `--workers 1` path retains serial behavior, avoiding process-startup overhead.
- With `--workers > 1`, `ProcessPoolExecutor.map()` dispatches full games and yields results in **game-index order**, even when jobs finish out of order.
- Per-game re-seeding and round-specific deterministic dealing preserve the original two-game paired deal/seat-swap design.
- Results, seat assignments, scores, margins, winners and CSV rows match the serial version **exactly** for fixed seeds. The `seconds` field differs because execution speed depends on CPU scheduling.
- `wall_seconds` reports **actual elapsed tournament duration**, while `total_seconds` is the sum of each game duration; the latter can be much larger when processes overlap.
- Tests compare four complete matches in one- and two-worker modes. The CI benchmark independently compares **all 12 game result rows** in one-, two- and four-worker modes, ignoring only timing fields.

## Measured speedup

| Concurrent **games** | Elapsed wall time | Speedup vs serial | Sum of individual game times |
| ---: | ---: | ---: | ---: |
| **1** | **48.109 seconds** | 1.00× | 48.098 s |
| **2** | **24.405 seconds** | **1.97×** | 48.769 s |
| **4** | **18.400 seconds** | **2.61×** | 71.015 s |

CI confirmation: `PARALLEL_SCORE_EQUALITY confirmed games=12 workers=1,2,4`.

**Identical playing results for every worker count:** 7 head-to-head wins for new trump choice, 5 for legacy; mean new score 102.667, legacy 95.417; mean new-minus-old margin +7.250. The benchmark was a CPU scheduling comparison, **not** intended as additional independent evidence for either trump algorithm's playing strength.

Four concurrent processes were faster than two, but scaling was less than linear (4× processes gave 2.61× speedup). In particular, the four-process run reported 71.015 cumulative game-seconds versus 48.098 serial game-seconds, reflecting contention/process overhead despite the shorter wall time. On a 2-core computer, running four games concurrently might be no faster than two.

## Estimated runtime for larger tournaments

From this 12-game workload, approximately **4.0 seconds/game sequentially** or about **1.53 seconds/game elapsed with four concurrent game processes** on this specific runner. Extrapolating:

| Games | Serial estimate | Four-worker estimate |
| ---: | ---: | ---: |
| 200 | ~13.4 minutes | ~5.1 minutes |
| 500 | ~33.4 minutes | ~12.8 minutes |

The preceding **50-game** experiment required 254.2 seconds (5.08 seconds/game) on a different shared runner. Extrapolating from that slower workload and the observed 2.61× speedup instead gives roughly **42.4 minutes serial / 16.2 minutes with four workers** for 500 games. Therefore roughly **13–16+ minutes for 500** is a planning estimate under similar hardware; it's not a guarantee, and larger variation is possible.

## Usage

```bash
python -m benchmarks.trump_tournament --games 50 --rollouts 1024 --workers 4 --csv 50-games.csv
python -m benchmarks.trump_tournament --games 500 --rollouts 1024 --workers 4 --csv 500-games.csv
```

For the apples-to-apples comparison:

```bash
python -m benchmarks.trump_tournament --games 12 --rollouts 1024 --seed 61234 --workers 1
python -m benchmarks.trump_tournament --games 12 --rollouts 1024 --seed 61234 --workers 2
python -m benchmarks.trump_tournament --games 12 --rollouts 1024 --seed 61234 --workers 4
```

**Important:** The default remains `--workers 1`, so existing scripts are unaffected. `--workers` controls **concurrent whole matches**; `monteCarloWorkers` still controls the separate, optional per-decision rollout parallelism. Keeping `monteCarloWorkers=1` inside each game avoids nested process pools and unnecessary oversubscription.

The one-time 12-game scaling experiment was run in the PR and then removed from routine CI to avoid multiplying the cost of every push. Its results are preserved above and in the linked historical CI logs.
