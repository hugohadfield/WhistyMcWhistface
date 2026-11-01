# Expected-score versus heuristic trump picker: 250-game, four-worker tournament

**Experiment date:** 1 November 2026 (UTC), 31 October 2026 (Pacific).  
**Game:** [WhistyMcWhistface](../README.md), three-player 13-round contract Whist.  
**Code revision:** `810c5e2af2e87d9f057f51a744dba915d5db7182` (unchanged implementation from PR #18).  
**Seed range:** `20261056` to `20261180` inclusive: **125 new deal-pair seeds**, avoiding the 25 used in the earlier 50-game experiment.  
**Configuration:** 250 complete games, **four concurrent whole-game processes**, **1,024 Monte Carlo rollouts per non-forced card decision**, one rollout worker per process.  
**Evidence:** [successful 250-game CI run #41285517515](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41285517515); [raw CSV artifact](https://github.com/hugohadfield/WhistyMcWhistface/actions/runs/41285517515/artifacts/13075366955) (GitHub retains this temporary artifact for 14 days; all 250 match records are also preserved below).  
**Command:** `python -m benchmarks.trump_tournament --games 250 --workers 4 --rollouts 1024 --seed 20261056 --csv /tmp/trump-tournament-250.csv`

## 1. Outcome

The tournament completed successfully. It compares the **new exact-DP expected-score trump selector** (active in MonteCarloPlayer) against **Ed Dakin's original hand-statistics heuristic** (used by the `HeuristicTrumpMonteCarloPlayer` control). The players are identical except for trump choice. A `RandomPlayer` fills the remaining seat.

| Measured quantity | New expected-score trump picker | Legacy heuristic |
| --- | ---: | ---: |
| Head-to-head wins out of 250 | **130 (52.0%)** | **118 (47.2%)** |
| Head-to-head ties | 2 (0.8%) | 2 (0.8%) |
| Outright first place out of all three players | **127 (50.8%)** | **117 (46.8%)** |
| Average cumulative final score | 93.336 | **93.656** |

The random third player won four games outright. Two games ended in a tie for first.

**The new picker won more head-to-head games, but scored slightly fewer points on average.** Its mean game-score margin (new minus legacy) was **−0.320 points**; the median was **+1 point**. Neither statistic establishes an overall advantage.

## 2. Paired-deal uncertainty

Each of 125 random deal seeds was played **twice**. The new and old agents swapped seats while the third random agent kept its seat, and the same initial hands were dealt in each round of each pair. This controls much of the hand and seat variance, although stochastic decisions and actions can diverge after the trump picks. For statistical uncertainty, each **two-game deal pair** is one unit rather than counting the 250 individual games as independent samples.

- **61/125 pairs** favored the new algorithm based on their two-game average score margin.
- **58/125 pairs** favored the old algorithm.
- **6/125 pairs** tied.
- Mean pair score difference: **−0.320 points per game**.
- Estimated paired bootstrap **95% confidence interval: −2.464 to +1.808 points/game**.

The percentile bootstrap resampled the **125 complete deal pairs** with replacement, 20,000 times, using a deterministic simple linear-congruential pseudorandom generator initialized to 991273, and took the 2.5th and 97.5th percentiles. This is a descriptive interval based on the tested pair distribution; it is not a proof of equivalence or a claim that either algorithm has positive expected advantage.

The interval includes zero and has both negative and positive values. **There is no persuasive evidence here that the new trump picker improves average score.** The slight 130-to-118 win difference is likewise not conclusive.

## 3. Combining the new data with the earlier 50-game tournament

The first 50-game experiment used 25 independent paired seeds, `20261031`–`20261055`; this 250-game follow-up uses **different seeds**, so together they cover **150 unique paired deals / 300 full games**.

| Quantity | First 50 | New 250 | Combined 300 |
| --- | ---: | ---: | ---: |
| New picker head-to-head wins | 30 | 130 | **160** |
| Legacy head-to-head wins | 19 | 118 | **137** |
| Ties | 1 | 2 | **3** |
| New minus old average final score | **+3.26** | **−0.32** | **+0.277** |
| Paired units favoring new | 11 | 61 | **72** |
| Paired units favoring legacy | 13 | 58 | **71** |
| Tied paired units | 1 | 6 | **7** |

Across the 150 paired units, the combined mean margin is **+0.277 points/game**; the same 20,000-resample paired bootstrap gives an approximate **95% interval of −1.733 to +2.283 points/game**. The earlier positive mean was not sustained in the larger independent sample. On these 300 games, the two trump policies look **closely matched**, and additional games would be needed to distinguish a small effect.

## 4. Execution and performance

The CI runner reported four available CPUs. The experiment took **378.565 seconds (6 minutes, 18.6 seconds)** of actual elapsed tournament time. The sum of 250 per-game durations was **1,505.983 seconds** because games ran concurrently and shared finite CPU resources. Do **not** treat that summed duration as serial elapsed runtime: it is process-observed game durations under parallel scheduling.

Every game completed the full **13 rounds**, **55 tricks**, and **165 actual card plays**, a total of **3,250 rounds, 13,750 tricks and 41,250 card plays** across this experiment. The full CI succeeded; **432 pytest tests passed** and the standard throughput, full-match, and convergence benchmarks also ran.

The 250-game run used the existing separate-process implementation and demonstrates the practicality of running many matches in parallel. The experiment does **not** independently measure serial versus parallel performance for these same 250 games; the [prior 12-game worker-scaling comparison](parallel-tournament-benchmark-2026-10-31.md) measured the speedup of parallelism directly.

## 5. Complete 250-game results

The following rows are extracted verbatim as numerical fields from the CI log; `new − old` is the two Monte Carlo players' final-score difference, and `winner` is the outright first-place player (or TIE if first place is shared). The CSV download provides identical data in machine-readable form.

| Game | Deal pair | Seed | New seat | Old seat | New score | Old score | Random score | New − old | Winner | Runtime (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 1 | 20261056 | 0 | 1 | 102 | 98 | 65 | +4 | NEW | 5.845 |
| 2 | 1 | 20261056 | 1 | 0 | 90 | 89 | 86 | +1 | NEW | 5.477 |
| 3 | 2 | 20261057 | 0 | 2 | 84 | 105 | 46 | -21 | OLD | 5.584 |
| 4 | 2 | 20261057 | 2 | 0 | 96 | 89 | 60 | +7 | NEW | 5.529 |
| 5 | 3 | 20261058 | 1 | 2 | 92 | 73 | 100 | +19 | RANDOM | 5.838 |
| 6 | 3 | 20261058 | 2 | 1 | 84 | 83 | 78 | +1 | NEW | 6.112 |
| 7 | 4 | 20261059 | 0 | 1 | 119 | 83 | 53 | +36 | NEW | 6.055 |
| 8 | 4 | 20261059 | 1 | 0 | 85 | 115 | 75 | -30 | OLD | 6.277 |
| 9 | 5 | 20261060 | 0 | 2 | 123 | 93 | 69 | +30 | NEW | 6.251 |
| 10 | 5 | 20261060 | 2 | 0 | 92 | 136 | 37 | -44 | OLD | 6.165 |
| 11 | 6 | 20261061 | 1 | 2 | 80 | 100 | 35 | -20 | OLD | 6.338 |
| 12 | 6 | 20261061 | 2 | 1 | 74 | 76 | 45 | -2 | OLD | 6.150 |
| 13 | 7 | 20261062 | 0 | 1 | 96 | 93 | 56 | +3 | NEW | 5.870 |
| 14 | 7 | 20261062 | 1 | 0 | 103 | 95 | 47 | +8 | NEW | 5.677 |
| 15 | 8 | 20261063 | 0 | 2 | 115 | 98 | 42 | +17 | NEW | 5.897 |
| 16 | 8 | 20261063 | 2 | 0 | 88 | 112 | 45 | -24 | OLD | 6.199 |
| 17 | 9 | 20261064 | 1 | 2 | 98 | 86 | 91 | +12 | NEW | 6.866 |
| 18 | 9 | 20261064 | 2 | 1 | 95 | 110 | 80 | -15 | OLD | 6.635 |
| 19 | 10 | 20261065 | 0 | 1 | 74 | 115 | 56 | -41 | OLD | 6.249 |
| 20 | 10 | 20261065 | 1 | 0 | 95 | 66 | 54 | +29 | NEW | 6.028 |
| 21 | 11 | 20261066 | 0 | 2 | 87 | 119 | 49 | -32 | OLD | 6.172 |
| 22 | 11 | 20261066 | 2 | 0 | 129 | 98 | 68 | +31 | NEW | 6.232 |
| 23 | 12 | 20261067 | 1 | 2 | 82 | 112 | 71 | -30 | OLD | 6.350 |
| 24 | 12 | 20261067 | 2 | 1 | 109 | 66 | 60 | +43 | NEW | 6.451 |
| 25 | 13 | 20261068 | 0 | 1 | 88 | 63 | 64 | +25 | NEW | 5.565 |
| 26 | 13 | 20261068 | 1 | 0 | 83 | 79 | 63 | +4 | NEW | 5.507 |
| 27 | 14 | 20261069 | 0 | 2 | 108 | 77 | 50 | +31 | NEW | 5.857 |
| 28 | 14 | 20261069 | 2 | 0 | 79 | 95 | 61 | -16 | OLD | 5.901 |
| 29 | 15 | 20261070 | 1 | 2 | 101 | 87 | 27 | +14 | NEW | 6.355 |
| 30 | 15 | 20261070 | 2 | 1 | 90 | 102 | 23 | -12 | OLD | 6.363 |
| 31 | 16 | 20261071 | 0 | 1 | 99 | 122 | 64 | -23 | OLD | 6.345 |
| 32 | 16 | 20261071 | 1 | 0 | 112 | 117 | 66 | -5 | OLD | 5.926 |
| 33 | 17 | 20261072 | 0 | 2 | 100 | 94 | 71 | +6 | NEW | 6.373 |
| 34 | 17 | 20261072 | 2 | 0 | 82 | 66 | 47 | +16 | NEW | 6.752 |
| 35 | 18 | 20261073 | 1 | 2 | 129 | 79 | 67 | +50 | NEW | 5.720 |
| 36 | 18 | 20261073 | 2 | 1 | 79 | 110 | 46 | -31 | OLD | 6.108 |
| 37 | 19 | 20261074 | 0 | 1 | 99 | 102 | 64 | -3 | OLD | 5.919 |
| 38 | 19 | 20261074 | 1 | 0 | 104 | 114 | 27 | -10 | OLD | 6.039 |
| 39 | 20 | 20261075 | 0 | 2 | 85 | 79 | 61 | +6 | NEW | 6.343 |
| 40 | 20 | 20261075 | 2 | 0 | 88 | 100 | 47 | -12 | OLD | 5.888 |
| 41 | 21 | 20261076 | 1 | 2 | 65 | 88 | 62 | -23 | OLD | 6.168 |
| 42 | 21 | 20261076 | 2 | 1 | 87 | 67 | 51 | +20 | NEW | 6.222 |
| 43 | 22 | 20261077 | 0 | 1 | 64 | 132 | 49 | -68 | OLD | 6.318 |
| 44 | 22 | 20261077 | 1 | 0 | 126 | 107 | 42 | +19 | NEW | 5.983 |
| 45 | 23 | 20261078 | 0 | 2 | 125 | 89 | 81 | +36 | NEW | 5.969 |
| 46 | 23 | 20261078 | 2 | 0 | 89 | 90 | 76 | -1 | OLD | 5.749 |
| 47 | 24 | 20261079 | 1 | 2 | 45 | 95 | 45 | -50 | OLD | 6.675 |
| 48 | 24 | 20261079 | 2 | 1 | 93 | 46 | 76 | +47 | NEW | 6.349 |
| 49 | 25 | 20261080 | 0 | 1 | 82 | 86 | 77 | -4 | OLD | 5.988 |
| 50 | 25 | 20261080 | 1 | 0 | 102 | 90 | 83 | +12 | NEW | 5.885 |
| 51 | 26 | 20261081 | 0 | 2 | 112 | 87 | 56 | +25 | NEW | 5.843 |
| 52 | 26 | 20261081 | 2 | 0 | 118 | 113 | 54 | +5 | NEW | 5.999 |
| 53 | 27 | 20261082 | 1 | 2 | 92 | 78 | 45 | +14 | NEW | 5.582 |
| 54 | 27 | 20261082 | 2 | 1 | 71 | 88 | 56 | -17 | OLD | 5.926 |
| 55 | 28 | 20261083 | 0 | 1 | 101 | 77 | 57 | +24 | NEW | 5.514 |
| 56 | 28 | 20261083 | 1 | 0 | 78 | 133 | 44 | -55 | OLD | 5.932 |
| 57 | 29 | 20261084 | 0 | 2 | 84 | 83 | 38 | +1 | NEW | 5.476 |
| 58 | 29 | 20261084 | 2 | 0 | 61 | 56 | 58 | +5 | NEW | 5.532 |
| 59 | 30 | 20261085 | 1 | 2 | 109 | 97 | 59 | +12 | NEW | 6.071 |
| 60 | 30 | 20261085 | 2 | 1 | 61 | 110 | 84 | -49 | OLD | 6.203 |
| 61 | 31 | 20261086 | 0 | 1 | 86 | 91 | 78 | -5 | OLD | 5.558 |
| 62 | 31 | 20261086 | 1 | 0 | 80 | 98 | 87 | -18 | OLD | 5.704 |
| 63 | 32 | 20261087 | 0 | 2 | 86 | 76 | 63 | +10 | NEW | 6.067 |
| 64 | 32 | 20261087 | 2 | 0 | 84 | 75 | 66 | +9 | NEW | 5.533 |
| 65 | 33 | 20261088 | 1 | 2 | 70 | 84 | 61 | -14 | OLD | 6.134 |
| 66 | 33 | 20261088 | 2 | 1 | 96 | 68 | 61 | +28 | NEW | 6.157 |
| 67 | 34 | 20261089 | 0 | 1 | 77 | 102 | 46 | -25 | OLD | 6.032 |
| 68 | 34 | 20261089 | 1 | 0 | 60 | 90 | 65 | -30 | OLD | 6.016 |
| 69 | 35 | 20261090 | 0 | 2 | 89 | 76 | 40 | +13 | NEW | 5.644 |
| 70 | 35 | 20261090 | 2 | 0 | 68 | 81 | 76 | -13 | OLD | 5.902 |
| 71 | 36 | 20261091 | 1 | 2 | 101 | 120 | 64 | -19 | OLD | 6.134 |
| 72 | 36 | 20261091 | 2 | 1 | 111 | 89 | 65 | +22 | NEW | 6.093 |
| 73 | 37 | 20261092 | 0 | 1 | 99 | 83 | 63 | +16 | NEW | 6.225 |
| 74 | 37 | 20261092 | 1 | 0 | 83 | 87 | 65 | -4 | OLD | 6.007 |
| 75 | 38 | 20261093 | 0 | 2 | 92 | 87 | 66 | +5 | NEW | 5.856 |
| 76 | 38 | 20261093 | 2 | 0 | 111 | 105 | 49 | +6 | NEW | 5.654 |
| 77 | 39 | 20261094 | 1 | 2 | 82 | 74 | 79 | +8 | NEW | 6.576 |
| 78 | 39 | 20261094 | 2 | 1 | 90 | 98 | 37 | -8 | OLD | 6.531 |
| 79 | 40 | 20261095 | 0 | 1 | 107 | 101 | 67 | +6 | NEW | 5.787 |
| 80 | 40 | 20261095 | 1 | 0 | 95 | 118 | 62 | -23 | OLD | 5.708 |
| 81 | 41 | 20261096 | 0 | 2 | 135 | 92 | 48 | +43 | NEW | 5.491 |
| 82 | 41 | 20261096 | 2 | 0 | 122 | 115 | 58 | +7 | NEW | 5.758 |
| 83 | 42 | 20261097 | 1 | 2 | 103 | 93 | 39 | +10 | NEW | 5.854 |
| 84 | 42 | 20261097 | 2 | 1 | 93 | 98 | 54 | -5 | OLD | 5.744 |
| 85 | 43 | 20261098 | 0 | 1 | 78 | 129 | 48 | -51 | OLD | 5.377 |
| 86 | 43 | 20261098 | 1 | 0 | 97 | 61 | 77 | +36 | NEW | 5.433 |
| 87 | 44 | 20261099 | 0 | 2 | 71 | 99 | 95 | -28 | OLD | 5.611 |
| 88 | 44 | 20261099 | 2 | 0 | 106 | 96 | 63 | +10 | NEW | 5.938 |
| 89 | 45 | 20261100 | 1 | 2 | 102 | 96 | 67 | +6 | NEW | 6.365 |
| 90 | 45 | 20261100 | 2 | 1 | 96 | 74 | 65 | +22 | NEW | 6.115 |
| 91 | 46 | 20261101 | 0 | 1 | 50 | 134 | 51 | -84 | OLD | 5.896 |
| 92 | 46 | 20261101 | 1 | 0 | 113 | 49 | 63 | +64 | NEW | 5.865 |
| 93 | 47 | 20261102 | 0 | 2 | 96 | 86 | 63 | +10 | NEW | 5.768 |
| 94 | 47 | 20261102 | 2 | 0 | 95 | 107 | 63 | -12 | OLD | 5.558 |
| 95 | 48 | 20261103 | 1 | 2 | 106 | 110 | 59 | -4 | OLD | 6.026 |
| 96 | 48 | 20261103 | 2 | 1 | 116 | 98 | 71 | +18 | NEW | 5.870 |
| 97 | 49 | 20261104 | 0 | 1 | 93 | 98 | 84 | -5 | OLD | 6.430 |
| 98 | 49 | 20261104 | 1 | 0 | 108 | 80 | 87 | +28 | NEW | 6.182 |
| 99 | 50 | 20261105 | 0 | 2 | 89 | 76 | 50 | +13 | NEW | 5.724 |
| 100 | 50 | 20261105 | 2 | 0 | 80 | 87 | 78 | -7 | OLD | 5.596 |
| 101 | 51 | 20261106 | 1 | 2 | 137 | 87 | 61 | +50 | NEW | 6.330 |
| 102 | 51 | 20261106 | 2 | 1 | 76 | 130 | 79 | -54 | OLD | 5.996 |
| 103 | 52 | 20261107 | 0 | 1 | 125 | 89 | 41 | +36 | NEW | 6.613 |
| 104 | 52 | 20261107 | 1 | 0 | 86 | 124 | 35 | -38 | OLD | 6.321 |
| 105 | 53 | 20261108 | 0 | 2 | 96 | 100 | 39 | -4 | OLD | 7.092 |
| 106 | 53 | 20261108 | 2 | 0 | 117 | 82 | 36 | +35 | NEW | 6.905 |
| 107 | 54 | 20261109 | 1 | 2 | 74 | 80 | 81 | -6 | RANDOM | 5.677 |
| 108 | 54 | 20261109 | 2 | 1 | 79 | 85 | 81 | -6 | OLD | 5.669 |
| 109 | 55 | 20261110 | 0 | 1 | 100 | 108 | 77 | -8 | OLD | 6.297 |
| 110 | 55 | 20261110 | 1 | 0 | 99 | 105 | 81 | -6 | OLD | 6.146 |
| 111 | 56 | 20261111 | 0 | 2 | 108 | 79 | 58 | +29 | NEW | 5.779 |
| 112 | 56 | 20261111 | 2 | 0 | 101 | 67 | 77 | +34 | NEW | 5.077 |
| 113 | 57 | 20261112 | 1 | 2 | 117 | 84 | 54 | +33 | NEW | 6.023 |
| 114 | 57 | 20261112 | 2 | 1 | 98 | 116 | 41 | -18 | OLD | 6.142 |
| 115 | 58 | 20261113 | 0 | 1 | 79 | 74 | 52 | +5 | NEW | 5.884 |
| 116 | 58 | 20261113 | 1 | 0 | 113 | 67 | 85 | +46 | NEW | 5.976 |
| 117 | 59 | 20261114 | 0 | 2 | 73 | 118 | 84 | -45 | OLD | 5.887 |
| 118 | 59 | 20261114 | 2 | 0 | 113 | 105 | 67 | +8 | NEW | 5.929 |
| 119 | 60 | 20261115 | 1 | 2 | 76 | 88 | 61 | -12 | OLD | 6.240 |
| 120 | 60 | 20261115 | 2 | 1 | 65 | 78 | 52 | -13 | OLD | 6.428 |
| 121 | 61 | 20261116 | 0 | 1 | 91 | 125 | 29 | -34 | OLD | 5.892 |
| 122 | 61 | 20261116 | 1 | 0 | 92 | 94 | 29 | -2 | OLD | 6.085 |
| 123 | 62 | 20261117 | 0 | 2 | 59 | 111 | 45 | -52 | OLD | 5.931 |
| 124 | 62 | 20261117 | 2 | 0 | 120 | 68 | 47 | +52 | NEW | 5.982 |
| 125 | 63 | 20261118 | 1 | 2 | 85 | 131 | 69 | -46 | OLD | 5.493 |
| 126 | 63 | 20261118 | 2 | 1 | 88 | 96 | 91 | -8 | OLD | 5.549 |
| 127 | 64 | 20261119 | 0 | 1 | 66 | 121 | 78 | -55 | OLD | 6.009 |
| 128 | 64 | 20261119 | 1 | 0 | 102 | 74 | 89 | +28 | NEW | 5.872 |
| 129 | 65 | 20261120 | 0 | 2 | 96 | 83 | 36 | +13 | NEW | 6.209 |
| 130 | 65 | 20261120 | 2 | 0 | 93 | 86 | 36 | +7 | NEW | 5.981 |
| 131 | 66 | 20261121 | 1 | 2 | 87 | 87 | 51 | 0 | TIE | 5.943 |
| 132 | 66 | 20261121 | 2 | 1 | 104 | 80 | 51 | +24 | NEW | 5.993 |
| 133 | 67 | 20261122 | 0 | 1 | 109 | 87 | 69 | +22 | NEW | 6.300 |
| 134 | 67 | 20261122 | 1 | 0 | 107 | 110 | 48 | -3 | OLD | 6.140 |
| 135 | 68 | 20261123 | 0 | 2 | 111 | 105 | 69 | +6 | NEW | 5.575 |
| 136 | 68 | 20261123 | 2 | 0 | 112 | 84 | 39 | +28 | NEW | 5.009 |
| 137 | 69 | 20261124 | 1 | 2 | 92 | 101 | 72 | -9 | OLD | 7.087 |
| 138 | 69 | 20261124 | 2 | 1 | 98 | 77 | 100 | +21 | RANDOM | 7.218 |
| 139 | 70 | 20261125 | 0 | 1 | 107 | 102 | 56 | +5 | NEW | 6.150 |
| 140 | 70 | 20261125 | 1 | 0 | 116 | 91 | 78 | +25 | NEW | 6.009 |
| 141 | 71 | 20261126 | 0 | 2 | 86 | 89 | 60 | -3 | OLD | 5.815 |
| 142 | 71 | 20261126 | 2 | 0 | 88 | 64 | 53 | +24 | NEW | 5.571 |
| 143 | 72 | 20261127 | 1 | 2 | 85 | 106 | 34 | -21 | OLD | 6.054 |
| 144 | 72 | 20261127 | 2 | 1 | 107 | 114 | 44 | -7 | OLD | 6.146 |
| 145 | 73 | 20261128 | 0 | 1 | 99 | 118 | 38 | -19 | OLD | 5.851 |
| 146 | 73 | 20261128 | 1 | 0 | 118 | 109 | 38 | +9 | NEW | 5.903 |
| 147 | 74 | 20261129 | 0 | 2 | 113 | 81 | 31 | +32 | NEW | 6.249 |
| 148 | 74 | 20261129 | 2 | 0 | 109 | 116 | 30 | -7 | OLD | 5.743 |
| 149 | 75 | 20261130 | 1 | 2 | 117 | 73 | 65 | +44 | NEW | 6.443 |
| 150 | 75 | 20261130 | 2 | 1 | 72 | 104 | 49 | -32 | OLD | 6.310 |
| 151 | 76 | 20261131 | 0 | 1 | 77 | 94 | 74 | -17 | OLD | 5.689 |
| 152 | 76 | 20261131 | 1 | 0 | 107 | 72 | 66 | +35 | NEW | 5.769 |
| 153 | 77 | 20261132 | 0 | 2 | 80 | 71 | 34 | +9 | NEW | 5.814 |
| 154 | 77 | 20261132 | 2 | 0 | 80 | 90 | 35 | -10 | OLD | 5.930 |
| 155 | 78 | 20261133 | 1 | 2 | 44 | 96 | 55 | -52 | OLD | 6.169 |
| 156 | 78 | 20261133 | 2 | 1 | 84 | 65 | 66 | +19 | NEW | 6.388 |
| 157 | 79 | 20261134 | 0 | 1 | 67 | 130 | 88 | -63 | OLD | 5.830 |
| 158 | 79 | 20261134 | 1 | 0 | 111 | 67 | 57 | +44 | NEW | 5.839 |
| 159 | 80 | 20261135 | 0 | 2 | 96 | 117 | 52 | -21 | OLD | 6.330 |
| 160 | 80 | 20261135 | 2 | 0 | 116 | 90 | 59 | +26 | NEW | 7.011 |
| 161 | 81 | 20261136 | 1 | 2 | 85 | 88 | 32 | -3 | OLD | 6.549 |
| 162 | 81 | 20261136 | 2 | 1 | 88 | 96 | 31 | -8 | OLD | 6.645 |
| 163 | 82 | 20261137 | 0 | 1 | 82 | 115 | 48 | -33 | OLD | 5.514 |
| 164 | 82 | 20261137 | 1 | 0 | 105 | 104 | 36 | +1 | NEW | 5.806 |
| 165 | 83 | 20261138 | 0 | 2 | 54 | 121 | 50 | -67 | OLD | 6.231 |
| 166 | 83 | 20261138 | 2 | 0 | 96 | 78 | 81 | +18 | NEW | 6.278 |
| 167 | 84 | 20261139 | 1 | 2 | 103 | 101 | 71 | +2 | NEW | 6.433 |
| 168 | 84 | 20261139 | 2 | 1 | 109 | 73 | 63 | +36 | NEW | 6.378 |
| 169 | 85 | 20261140 | 0 | 1 | 82 | 106 | 77 | -24 | OLD | 5.643 |
| 170 | 85 | 20261140 | 1 | 0 | 97 | 91 | 57 | +6 | NEW | 5.885 |
| 171 | 86 | 20261141 | 0 | 2 | 101 | 77 | 57 | +24 | NEW | 6.305 |
| 172 | 86 | 20261141 | 2 | 0 | 82 | 102 | 51 | -20 | OLD | 5.917 |
| 173 | 87 | 20261142 | 1 | 2 | 100 | 136 | 49 | -36 | OLD | 6.734 |
| 174 | 87 | 20261142 | 2 | 1 | 121 | 98 | 66 | +23 | NEW | 6.718 |
| 175 | 88 | 20261143 | 0 | 1 | 86 | 119 | 40 | -33 | OLD | 6.101 |
| 176 | 88 | 20261143 | 1 | 0 | 108 | 101 | 46 | +7 | NEW | 5.925 |
| 177 | 89 | 20261144 | 0 | 2 | 98 | 78 | 49 | +20 | NEW | 5.943 |
| 178 | 89 | 20261144 | 2 | 0 | 94 | 118 | 63 | -24 | OLD | 5.905 |
| 179 | 90 | 20261145 | 1 | 2 | 81 | 96 | 58 | -15 | OLD | 6.919 |
| 180 | 90 | 20261145 | 2 | 1 | 105 | 90 | 50 | +15 | NEW | 6.248 |
| 181 | 91 | 20261146 | 0 | 1 | 76 | 75 | 44 | +1 | NEW | 6.051 |
| 182 | 91 | 20261146 | 1 | 0 | 88 | 97 | 30 | -9 | OLD | 6.131 |
| 183 | 92 | 20261147 | 0 | 2 | 82 | 99 | 84 | -17 | OLD | 5.563 |
| 184 | 92 | 20261147 | 2 | 0 | 121 | 62 | 102 | +59 | NEW | 5.943 |
| 185 | 93 | 20261148 | 1 | 2 | 109 | 112 | 44 | -3 | OLD | 6.050 |
| 186 | 93 | 20261148 | 2 | 1 | 102 | 98 | 45 | +4 | NEW | 6.071 |
| 187 | 94 | 20261149 | 0 | 1 | 88 | 78 | 69 | +10 | NEW | 6.599 |
| 188 | 94 | 20261149 | 1 | 0 | 95 | 105 | 65 | -10 | OLD | 6.387 |
| 189 | 95 | 20261150 | 0 | 2 | 100 | 90 | 55 | +10 | NEW | 6.335 |
| 190 | 95 | 20261150 | 2 | 0 | 76 | 102 | 57 | -26 | OLD | 6.025 |
| 191 | 96 | 20261151 | 1 | 2 | 118 | 109 | 38 | +9 | NEW | 5.460 |
| 192 | 96 | 20261151 | 2 | 1 | 108 | 116 | 41 | -8 | OLD | 5.320 |
| 193 | 97 | 20261152 | 0 | 1 | 108 | 123 | 44 | -15 | OLD | 5.944 |
| 194 | 97 | 20261152 | 1 | 0 | 120 | 104 | 61 | +16 | NEW | 5.774 |
| 195 | 98 | 20261153 | 0 | 2 | 76 | 104 | 35 | -28 | OLD | 6.293 |
| 196 | 98 | 20261153 | 2 | 0 | 93 | 82 | 60 | +11 | NEW | 6.064 |
| 197 | 99 | 20261154 | 1 | 2 | 57 | 69 | 59 | -12 | OLD | 5.928 |
| 198 | 99 | 20261154 | 2 | 1 | 91 | 58 | 76 | +33 | NEW | 6.276 |
| 199 | 100 | 20261155 | 0 | 1 | 100 | 64 | 71 | +36 | NEW | 6.260 |
| 200 | 100 | 20261155 | 1 | 0 | 102 | 91 | 82 | +11 | NEW | 5.922 |
| 201 | 101 | 20261156 | 0 | 2 | 81 | 107 | 57 | -26 | OLD | 5.815 |
| 202 | 101 | 20261156 | 2 | 0 | 117 | 90 | 48 | +27 | NEW | 5.730 |
| 203 | 102 | 20261157 | 1 | 2 | 90 | 112 | 63 | -22 | OLD | 6.209 |
| 204 | 102 | 20261157 | 2 | 1 | 100 | 99 | 86 | +1 | NEW | 6.306 |
| 205 | 103 | 20261158 | 0 | 1 | 103 | 64 | 68 | +39 | NEW | 6.350 |
| 206 | 103 | 20261158 | 1 | 0 | 72 | 87 | 66 | -15 | OLD | 6.444 |
| 207 | 104 | 20261159 | 0 | 2 | 89 | 120 | 76 | -31 | OLD | 6.145 |
| 208 | 104 | 20261159 | 2 | 0 | 121 | 86 | 88 | +35 | NEW | 6.483 |
| 209 | 105 | 20261160 | 1 | 2 | 114 | 94 | 47 | +20 | NEW | 6.025 |
| 210 | 105 | 20261160 | 2 | 1 | 104 | 104 | 47 | 0 | TIE | 6.347 |
| 211 | 106 | 20261161 | 0 | 1 | 137 | 105 | 43 | +32 | NEW | 5.979 |
| 212 | 106 | 20261161 | 1 | 0 | 114 | 101 | 40 | +13 | NEW | 5.677 |
| 213 | 107 | 20261162 | 0 | 2 | 119 | 111 | 25 | +8 | NEW | 6.079 |
| 214 | 107 | 20261162 | 2 | 0 | 76 | 141 | 48 | -65 | OLD | 5.904 |
| 215 | 108 | 20261163 | 1 | 2 | 89 | 106 | 40 | -17 | OLD | 6.059 |
| 216 | 108 | 20261163 | 2 | 1 | 112 | 93 | 30 | +19 | NEW | 6.366 |
| 217 | 109 | 20261164 | 0 | 1 | 80 | 98 | 37 | -18 | OLD | 5.677 |
| 218 | 109 | 20261164 | 1 | 0 | 74 | 82 | 49 | -8 | OLD | 5.729 |
| 219 | 110 | 20261165 | 0 | 2 | 89 | 61 | 85 | +28 | NEW | 5.862 |
| 220 | 110 | 20261165 | 2 | 0 | 52 | 109 | 74 | -57 | OLD | 5.686 |
| 221 | 111 | 20261166 | 1 | 2 | 101 | 81 | 43 | +20 | NEW | 6.363 |
| 222 | 111 | 20261166 | 2 | 1 | 71 | 91 | 43 | -20 | OLD | 6.143 |
| 223 | 112 | 20261167 | 0 | 1 | 74 | 78 | 73 | -4 | OLD | 5.886 |
| 224 | 112 | 20261167 | 1 | 0 | 98 | 87 | 40 | +11 | NEW | 5.586 |
| 225 | 113 | 20261168 | 0 | 2 | 79 | 109 | 77 | -30 | OLD | 6.383 |
| 226 | 113 | 20261168 | 2 | 0 | 86 | 85 | 64 | +1 | NEW | 5.757 |
| 227 | 114 | 20261169 | 1 | 2 | 95 | 93 | 57 | +2 | NEW | 6.329 |
| 228 | 114 | 20261169 | 2 | 1 | 101 | 70 | 34 | +31 | NEW | 6.426 |
| 229 | 115 | 20261170 | 0 | 1 | 120 | 89 | 76 | +31 | NEW | 6.183 |
| 230 | 115 | 20261170 | 1 | 0 | 74 | 112 | 69 | -38 | OLD | 6.075 |
| 231 | 116 | 20261171 | 0 | 2 | 97 | 67 | 31 | +30 | NEW | 6.397 |
| 232 | 116 | 20261171 | 2 | 0 | 86 | 103 | 36 | -17 | OLD | 6.159 |
| 233 | 117 | 20261172 | 1 | 2 | 96 | 60 | 19 | +36 | NEW | 5.886 |
| 234 | 117 | 20261172 | 2 | 1 | 57 | 90 | 28 | -33 | OLD | 5.999 |
| 235 | 118 | 20261173 | 0 | 1 | 52 | 101 | 52 | -49 | OLD | 6.048 |
| 236 | 118 | 20261173 | 1 | 0 | 80 | 83 | 52 | -3 | OLD | 6.398 |
| 237 | 119 | 20261174 | 0 | 2 | 82 | 87 | 76 | -5 | OLD | 6.022 |
| 238 | 119 | 20261174 | 2 | 0 | 105 | 70 | 100 | +35 | NEW | 5.638 |
| 239 | 120 | 20261175 | 1 | 2 | 61 | 108 | 76 | -47 | OLD | 5.602 |
| 240 | 120 | 20261175 | 2 | 1 | 124 | 93 | 48 | +31 | NEW | 5.739 |
| 241 | 121 | 20261176 | 0 | 1 | 99 | 109 | 47 | -10 | OLD | 6.880 |
| 242 | 121 | 20261176 | 1 | 0 | 106 | 98 | 51 | +8 | NEW | 6.847 |
| 243 | 122 | 20261177 | 0 | 2 | 58 | 98 | 79 | -40 | OLD | 6.062 |
| 244 | 122 | 20261177 | 2 | 0 | 85 | 61 | 89 | +24 | RANDOM | 5.744 |
| 245 | 123 | 20261178 | 1 | 2 | 103 | 116 | 46 | -13 | OLD | 6.125 |
| 246 | 123 | 20261178 | 2 | 1 | 94 | 74 | 87 | +20 | NEW | 6.362 |
| 247 | 124 | 20261179 | 0 | 1 | 67 | 109 | 69 | -42 | OLD | 5.772 |
| 248 | 124 | 20261179 | 1 | 0 | 100 | 57 | 48 | +43 | NEW | 5.646 |
| 249 | 125 | 20261180 | 0 | 2 | 63 | 109 | 103 | -46 | OLD | 3.914 |
| 250 | 125 | 20261180 | 2 | 0 | 78 | 96 | 81 | -18 | OLD | 4.063 |

## 6. Reproducing

~~~bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m benchmarks.trump_tournament \
    --games 250 --workers 4 --rollouts 1024 \
    --seed 20261056 --csv tournament-250.csv
~~~

The tournament runner preserves game ordering across concurrent worker counts. Because every match independently resets Python's random seed, game scores, winners, seating and paired deals should match for 1, 2, or 4 whole-game workers (apart from per-game wall-clock time). Automated regression tests and the earlier 12-game CPU-scaling experiment verify this determinism.

**Conclusion:** The larger 250-game experiment does **not** corroborate a meaningful performance advantage for the new trump-selection policy. It remains the active default because that was the requested implementation choice, but the original heuristic is preserved for comparison. It would be premature to claim that the new picker is stronger or to revert it solely from this experiment; further work could focus on stronger opponent models, controlling RNG streams, or on the cases where the policies choose different trumps.
