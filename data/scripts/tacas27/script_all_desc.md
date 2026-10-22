# `runme.sh` — all TACAS'27 benchmarks in one run

```
./runme.sh <action> <mode> [--resume] [-n] [-h]
```

Actions: `run`, `collect-stats` (more to come: `plot-performance`, `run-verix-plus`).
Modes: `smoke`, `quick`, `short`, `full`. Both are required.

`runme.sh` runs the `plain_itp` experiments with the `neuron_activations` variants on the
benchmarks heart_attack, obesity, loan (tabular) and mnist, cifar, gtsrb (images), for the 50x
and 200x networks, on the dataset `<name>_s100_scaled.csv` (100 samples each).

- **One job** = one model × one variant × one experiment = one `spexplain` process.
  Its samples are always explained **sequentially**.
- All jobs of all benchmarks and widths share **one GNU parallel queue**. At most
  `CPU_PERCENTAGE` (default 80%) of the CPUs are busy at any time, the same as one
  `run-experiments.sh` call, so the time per explanation is not affected by running
  benchmarks side by side.
- The longest jobs (by their worst case) start first, so they do not prolong the end of the run.
- Every mode runs on **its own dataset** (see "Mode datasets" below). The results go to
  `data/explanations/<name>/<width>x<depth>/<dataset>/<variant>/`, where `<dataset>` is
  `s100_scaled_smoke`, `s100_scaled_quick`, `s100_scaled_short` or `s100_scaled` (full), so
  the modes never overwrite each other's results.
- The job list and GNU parallel's joblog are in `data/logs/runme/<mode>/`.
- `-n` (or `DRY_RUN=1`) prints the job counts and the worst-case estimate without running anything.
- `--resume` skips the jobs that already finished in the previous run of the same mode, e.g.
  after hitting the 24h Slurm limit.

Variants: `naive`, `guide-all` (prefer-all), `fix-all`, `hybrid`.
Experiments: `itp_astronger_bstrong`, `itp_astrong_bstrong`, `itp_afactor_0.5_bstrong`,
`itp_aweak_bstrong`, `itp_aweaker_bstrong`.

Models: the script runs all `<name>_<width>x*.nnet` files. The numbers below are for the
**final model set**, where only these depths `d` (number of hidden layers) are kept, for both 50x and 200x:

| benchmark | depths d | 50x | 200x |
|---|---|---|---|
| heart_attack, obesity, loan (tabular) | 1, 2, 3, 4, 6, 8 | 6 each | 6 each |
| mnist, cifar, gtsrb (images) | 1, 2, 3, 4, 6, 8, 10, 12, 14 | 9 each | 9 each |
| **total** | | **45** | **45** |

Until the other model files are removed, restrict a run to the final set with

```bash
FILTER='^((heart_attack|obesity|loan)/(50|200)x(1|2|3|4|6|8)|(mnist|cifar|gtsrb)/(50|200)x(1|2|3|4|6|8|10|12|14))/' ./runme.sh run <mode> ...
```

## The four modes

| | `smoke` | `quick` | `short` | `full` |
|---|---|---|---|---|
| benchmarks | all but cifar, gtsrb | all | all | all |
| widths | 50x | 50x, 200x | 50x, 200x | 50x, 200x |
| variants | guide-all, fix-all, hybrid | all 4, minus the exclusions below | all 4, minus the exclusions below | all 4 |
| experiments | itp_aweak_bstrong | 50x: all 5; 200x: itp_aweak_bstrong | 50x: all 5; 200x: itp_aweak_bstrong | all 5 |
| dataset | `<name>_s100_scaled_smoke.csv` | `<name>_s100_scaled_quick.csv` | `<name>_s100_scaled_short.csv` | `<name>_s100_scaled.csv` |
| samples | the important ones (1–6) | 8 (same as short) | 8 (same as quick) | all 100 |
| timeout per explanation | 1m tabular, 5m mnist | 200s tabular, MN 300s, CIF 400s, GTS 600s | HA/OB/LO 5m, MN/CIF 10m, GTS 14m | HA/OB/LO 5m, MN/CIF 10m, GTS 14m |
| on a failed job | stop | continue | continue | continue |
| results in | `s100_scaled_smoke/<variant>/` | `s100_scaled_quick/<variant>/` | `s100_scaled_short/<variant>/` | `s100_scaled/<variant>/` |
| **jobs** | **81** | **947** | **947** | **1,800** |
| **explanations** | **189** | **7,576** | **7,576** | **180,000** |

`smoke`, `short` and `quick` leave out the deep models where a variant mostly times out
(for `smoke` this changes nothing yet: it runs no `naive` and no 200x jobs):

| variant | benchmarks | left out |
|---|---|---|
| naive | heart_attack, obesity, loan | 50x8 (and deeper) |
| naive | mnist | 50x6 and deeper, 200x3 and deeper |
| naive | cifar, gtsrb | 50x4 and deeper, 200x3 and deeper |
| guide-all (prefer-all) | cifar, gtsrb | 200x4 and deeper |

`quick` differs from `short` only in the lower timeouts: same samples, separate results folders.

## Mode datasets

`make_mode_datasets.py` makes the datasets of smoke, quick and short from
`data/datasets/<name>/<name>_s100_scaled.csv` (full uses that one directly). Sample numbers
are 1-based, as in the datasets and in spexplain's `sample [i/N]` output.

| benchmark | smoke | quick = short (8) |
|---|---|---|
| heart_attack | 21, 55, 63, 84, 90, 98 | 14, 21, 55, 62, 63, 84, 90, 98 |
| obesity | 77 | 2, 11, 24, 42, 45, 52, 81, 84 |
| loan | 8, 70 | 8, 11, 24, 37, 38, 70, 81, 90 |
| mnist | 1 | 1, 7, 11, 35, 41, 46, 84, 87 |
| cifar | – (skipped) | 21, 30, 33, 55, 65, 84, 94, 97 |
| gtsrb | – (skipped) | 21, 29, 44, 45, 49, 51, 59, 60 |

- **Important samples**, in every mode dataset: heart_attack 98, 63, 84, 55, 21, 90;
  loan 8, 70; mnist 1 and 7 (smoke: only 1). The other samples are random, with a fixed seed
  (`SEED` in the script), so re-running the script gives the same datasets.
- **Map** from the samples of the mode datasets to the original ones:
  `data/datasets/mode_sample_maps.csv`, columns `benchmark,mode,index,original_index`
  (full is included as the identity). Use it to relate e.g. sample 3 of the short results
  of heart_attack to its original sample 55.
- **Neuron activations**: the hybrid variant reads `data/neuron_activations/<name>/<model>.txt`,
  whose blocks are numbered by the samples of the dataset. With a smaller dataset, block 1
  would belong to the wrong sample, and spexplain would not notice. So the script also writes
  renumbered copies to `data/neuron_activations/<name>/s100_scaled_<mode>/<model>.txt`, which
  `run1.sh` uses for that dataset. They keep all blocks of each sample as in the original files
  (some list a sample twice, some lack samples), so hybrid behaves as on the full dataset.
  `run` stops if one of them is missing.
- After changing the important samples or adding models, run the script again:
  `python3 make_mode_datasets.py`. The results of the earlier mode datasets are then outdated.

Jobs per benchmark and width:

| benchmark | smoke | quick, short | full |
|---|---|---|---|
| heart_attack 50x | 18 | 115 | 120 |
| obesity 50x | 18 | 115 | 120 |
| loan 50x | 18 | 115 | 120 |
| mnist 50x | 27 | 155 | 180 |
| cifar 50x | – | 150 | 180 |
| gtsrb 50x | – | 150 | 180 |
| heart_attack 200x | – | 24 | 120 |
| obesity 200x | – | 24 | 120 |
| loan 200x | – | 24 | 120 |
| mnist 200x | – | 29 | 180 |
| cifar 200x | – | 23 | 180 |
| gtsrb 200x | – | 23 | 180 |

## Runtime

All numbers assume a machine with **16 CPUs** and the default `CPU_PERCENTAGE=80`, i.e.
**13 jobs at once**, and the final model set. For another machine, run e.g.
`CPUS=<n> ./runme.sh run short -n` (with the `FILTER` above until the models are removed).

**Max** = every explanation hits its timeout. This is a hard upper bound, apart from the
small model-loading overhead per job. The wall time simulates the shared queue (the
longest jobs first, each to the next free slot). These numbers are exact and printed by `-n`.

**Expected** is an estimate. The only timings available are for small heart_attack/loan
models, where `guide-all`, `fix-all` and `hybrid` take well below a second up to a few seconds
per sample, while `naive` takes tens of seconds and often reaches the timeout already for
50x3. There are no timings yet for mnist, cifar, gtsrb or any 200x model. "Naive-bound"
below assumes that every `naive` explanation times out and the other variants are fast,
which is the most likely bad case. Calibrate it with the joblog of the smoke run.

| | `smoke` | `quick` | `short` | `full` |
|---|---|---|---|---|
| worst-case CPU time | 4h 57m | 692h | 1,077h | 26,400h |
| longest single job (max) | 6m | 1h 20m | 1h 52m | 23h 20m |
| **max wall time** | **23m** | **53h 20m (~2.2 days)** | **82h 56m (~3.5 days)** | **2,036h 40m (~85 days)** |
| naive-bound wall time | – (no naive) | 7h 20m | 11h 44m | 513h 20m (~21 days) |
| **expected** | **~5–10m** | **~4–8h** | **~6–12h** | **~3 weeks** |

Notes:
- `smoke` skips cifar and gtsrb and has no `naive` jobs; its 81 jobs explain 1–6 samples each
  (heart_attack 6, loan 2, mnist and obesity 1). Even if every explanation times out, it
  finishes in 23m.
- `short` and `quick`: with 13 slots the queue is always full, so their wall time is about
  the CPU time divided by 13. The exclusions remove the deep `naive` jobs, so the remaining
  `naive` jobs (149, up to 146h CPU time in short) are no longer most of the worst case:
  that is now the other variants on the large image models (798 jobs, up to 931h in short).
  Whether those time out is unknown until the smoke run.
- `full` is not practical on 16 CPUs: even the naive-bound case takes about 3 weeks.
  It needs a cluster node (e.g. 128 jobs at once: naive-bound ~2.5 days) or the
  reductions below. Under Slurm, resubmit the same command with `--resume` after the 24h
  limit; a job killed by the limit starts from scratch on resume.
- With 80%, 13 jobs run on 16 CPUs. If these are 8 physical cores with hyperthreading,
  some jobs share a core and run slower than one job per core, which also means more timeouts.

### Effect of the exclusions, 8 samples and the quick timeouts

Compared with the earlier `short` (first 10 samples, no exclusions), same 16 CPUs:

| | short before (10 samples) | short now | quick |
|---|---|---|---|
| jobs | 1,080 | 947 | 947 |
| explanations | 10,800 | 7,576 | 7,576 |
| longest job | 2h 20m | 1h 52m | 1h 20m |
| naive-bound | 31h | 11h 44m | 7h 20m |
| max (everything times out) | 122h | 83h | 53h |
| expected | ~1–1.5 days | ~6–12h | ~4–8h |

The exclusions remove the deep `naive` jobs, which did most of the work (`naive` CPU time
in short: 396h → 146h). Eight samples instead of ten cut another 20% from everything,
and the lower timeouts of `quick` cut about 35% more.

### Shared queue vs. one call per benchmark and width

Before, `run-experiments.sh` was called once per benchmark and width, one call after another,
and each call waited for its slowest job. Now all jobs share one queue. The time per
explanation is the same; only the idle cores at the end of each call are gone.
16 CPUs, 80% → 13 jobs at once:

| | one call per benchmark/width (before) | one shared queue (now) | saved |
|---|---|---|---|
| `quick`, worst case | 55h 45m | 53h 20m | ~4% |
| `quick`, only naive times out | 12h 36m | 7h 20m | ~42% |
| `short`, worst case | 86h 48m | 82h 56m | ~4% |
| `short`, only naive times out | 19h 36m | 11h 44m | ~40% |
| `full`, worst case | 2,086h 40m | 2,036h 40m | ~2% |
| `full`, only naive times out | 603h 20m | 513h 20m | ~15% |

The shared queue helps when there are more free slots than jobs at the end of a call, which
is rare with 13 slots, except when only a few long `naive` jobs are left. On a 128-job cluster node, it saves 40–80% for `short`.

## Examples

```bash
# check what would run on a 16-CPU machine
CPUS=16 ./runme.sh run short -n

# smoke test locally (needs bash >= 4, e.g. Homebrew's)
/opt/homebrew/bin/bash ./runme.sh run smoke

# full run on Eiger; resubmit with --resume after the 24h limit
WRAPPER=~/srun_commands/sbatch.sh ./runme.sh run full
WRAPPER=~/srun_commands/sbatch.sh ./runme.sh run full --resume

# only some jobs, e.g. rerun mnist 200x hybrid
FILTER='^mnist/200x[0-9]+/hybrid/' ./runme.sh run full
```

## Collecting the stats

```bash
./runme.sh collect-stats <mode> [-n]
```

`collect-stats` runs `collect_stats.sh` on the results of the given mode and writes one file
per benchmark, with both widths, to `./stats/<mode>/<name>.txt`, e.g. `./stats/short/cifar.txt`
(`STATS_DIR` changes `./stats`).

- It uses the same job list as `run`, so it only collects what the mode runs: the right
  results folder (`s100_scaled_<mode>/` or `s100_scaled/` for full), only the
  experiments of the mode, and none of the excluded jobs. `FILTER` works the same way.
- Each table is preceded by a header line, e.g. `cifar 50x1 naive`, as in `collect_all_stats.sh`.
- The variant directories are processed in parallel (same number of jobs as `run`).
- Every experiment is collected on its own, so all finished experiments always appear, also
  while others are still running or missing. They are merged into one table per variant
  directory; the others follow the table as `Not finished: <variant>/<experiment>`.
  An experiment is finished when its `.time.txt` has been written (at the end of the run).
- An experiment whose stats `collect_stats.sh` rejects (e.g. an inconsistency) is listed as
  `Failed: ...` with the message, and the action exits with 1. Not finished experiments
  do not change the exit status.
- `-n` lists the stats files and variant directories without collecting anything.

`collect_stats.sh` can also be used directly on a mode's results, without `<max_samples>`, e.g.
`./collect_stats.sh explanations/heart_attack/50x1/s100_scaled_short/hybrid/ plain_itp`.

## Files

- `runme.sh`: the actions, the job list and the GNU parallel queue.
- `modes.sh`: what each mode runs: benchmarks, widths, variants, experiments, exclusions,
  datasets and timeouts.
- `make_mode_datasets.py`: the mode datasets, the sample maps and their neuron activations.


## Run Times on My Laptop
- CPU: Apple M1 Pro, 10 cores (8 performance, 2 efficiency)
- Smoke: 27 minutes, 30 seconds
