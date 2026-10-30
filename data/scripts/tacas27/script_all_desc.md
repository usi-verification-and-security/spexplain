````# `runme.sh` — all TACAS'27 benchmarks in one run

```
./runme.sh <action> <mode> [--resume] [-n] [-h]
```

Actions: `run`, `collect-stats`, `plot-performance`, `interpret`, `run-verix-plus`.
Modes: `smoke`, `short`, `long`, `full`. Both are required.

`runme.sh` runs the `plain_itp` experiments with the `tacas27` variants on the
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
  `s100_scaled_smoke`, `s100_scaled_short`, `s100_scaled_long` or `s100_scaled` (full), so
  the modes never overwrite each other's results.
- The job list and GNU parallel's joblog are in `data/logs/runme/<mode>/`.
- `-n` (or `DRY_RUN=1`) prints the job counts and the worst-case estimate without running anything.
- `--resume` skips the jobs that already finished in the previous run of the same mode, e.g.
  after hitting the 24h Slurm limit.

Variants (`data/scripts/spec/variants/tacas27`): `naive`, `guide-all` (prefer-all), `fix-all`, `hybrid`,
`fix-all-no-vars` (fix-all without neuron variables in the explanations).
Experiments: `itp_astronger_bstrong`, `itp_astrong_bstrong`, `itp_afactor_0.5_bstrong`,
`itp_aweak_bstrong`, `itp_aweaker_bstrong`. In every mode but `smoke` also `ucore_min_itp_aweak_bstrong`, only for
heart_attack 50x1 `naive` and `fix-all` (see "Unsat cores of interpolants" below).

Models: only these `<name>_<width>x<depth>.nnet` files, for both widths 50x and 200x
(`TABULAR_DEPTHS` and `IMAGE_DEPTHS` in `modes.sh`; a missing one stops the script):

| benchmark | depths d | 50x | 200x |
|---|---|---|---|
| heart_attack, obesity, loan (tabular) | 1, 2, 3, 4, 6, 8 | 6 each | 6 each |
| mnist, cifar, gtsrb (images) | 1, 2, 3, 4, 6, 8, 10, 12, 14 | 9 each | 9 each |
| **total** | | **45** | **45** |

## The modes

| | `smoke` | `short` | `long` | `full` |
|---|---|---|---|---|
| benchmarks | all but cifar, gtsrb | all | all | all |
| widths | 50x, up to 50x6 | 50x, 200x | 50x, 200x | 50x, 200x |
| variants | guide-all, fix-all, hybrid, fix-all-no-vars | all 5, minus the exclusions below | all 5, minus the exclusions below | all 5, minus the fix-all-no-vars exclusions |
| experiments | itp_aweak_bstrong | 50x: all 5; 200x: itp_aweak_bstrong | 50x: all 5; 200x: itp_aweak_bstrong | all 5 |
| dataset | `<name>_s100_scaled_smoke.csv` | `<name>_s100_scaled_short.csv` | `<name>_s100_scaled_long.csv` | `<name>_s100_scaled.csv` |
| samples | the important ones (1–6) | 8 (the first 8 of long) | 20 (those of short, then 12 more) | all 100 |
| timeout per explanation | 1m tabular, 5m mnist | 200s tabular, MN 300s, CIF 400s, GTS 600s | HA/OB/LO 5m, MN/CIF 10m, GTS 14m | HA/OB/LO 5m, MN/CIF 10m, GTS 14m |
| on a failed job | stop | continue | continue | continue |
| results in | `s100_scaled_smoke/<variant>/` | `s100_scaled_short/<variant>/` | `s100_scaled_long/<variant>/` | `s100_scaled/<variant>/` |
| **jobs** (FC + CNN) | **90** (72 + 18) | **1,153** (1,009 + 144) | **1,297** (1,009 + 288) | **2,150** (1,862 + 288) |
| **explanations** (FC + CNN) | **204** (186 + 18) | **9,224** (8,072 + 1,152) | **25,940** (20,180 + 5,760) | **215,000** (186,200 + 28,800) |

`fix-all-no-vars` runs in all modes, but only for the 50x1–50x4 models of heart_attack, loan and mnist.
`smoke` runs only the models up to 50x6, no `naive` and no unsat-core jobs.

`smoke`, `short` and `long` also leave out the deep models where a variant mostly times out
(for `smoke` these change nothing beyond that limit, as it runs no 200x jobs):

| variant | benchmarks | left out |
|---|---|---|
| naive | heart_attack, obesity, loan | 50x8 (and deeper) |
| naive | mnist | 50x6 and deeper, 200x3 and deeper |
| naive | cifar, gtsrb | 50x4 and deeper, 200x3 and deeper |
| guide-all (prefer-all) | cifar, gtsrb | 200x4 and deeper |

`short` differs from `long` in the lower timeouts and in its samples: the first 8 of long
(the same jobs, separate results folders).

## CNN (ONNX) models

Every mode also runs the CNN-bench models of `data/scripts/cnn/tacas27/{MN,CIF,GTS}.sh`, as jobs of
the same GNU parallel queue (`CNN_*`, `set_cnn_mode` and `is_cnn_excluded` in `modes.sh`):

| | `smoke` | `short` | `long`, `full` |
|---|---|---|---|
| benchmarks | mnist | mnist, cifar, gtsrb | mnist, cifar, gtsrb |
| models (`data/models/<name>/cnn-bench/`) | NS1–NS3, NAB1–NAB3 | NS1–NS6, NAB1–NAB6 | NS1–NS6, NAB1–NAB6, S1–S6, AB1–AB6 |
| variants (`data/scripts/cnn/spec/variants/all`) | guide-all, fix-all, hybrid | naive, guide-all, fix-all, hybrid | naive, guide-all, fix-all, hybrid |
| experiments | itp_aweak_bstrong | itp_aweak_bstrong | itp_aweak_bstrong |
| **jobs** | **18** | **144** (48 per benchmark) | **288** (96 per benchmark) |

The families: NS (narrow, stride 2), NAB (narrow, average pooling), S (stride 2) and AB
(average pooling); S and AB have about twice the ReLUs of NS and NAB at the same depth
(`manifest.csv`). S and AB run only in `long` and `full` (`is_cnn_excluded`).

- The dataset and the timeout per explanation are those of the mode for the benchmark, as for the
  `.nnet` models (full: MN/CIF 10m, GTS 14m, as in `MN.sh`, `CIF.sh`, `GTS.sh`). The input domain
  is [0,1] (`CNN_INPUT_MIN`, `CNN_INPUT_MAX`).
- A job runs `data/scripts/cnn/run1.sh`, i.e. the frozen engine for ONNX networks
  `build-onnx/engine` with the `explain-onnx` action. Make it first with `make onnx`;
  `run` stops if it is missing (`-n` only warns).
- Job ids are `<name>/<id>/<variant>/<experiment>`, e.g. `mnist/NAB1/hybrid/itp_aweak_bstrong`,
  so `FILTER='^mnist/NAB'` selects them. In the estimate, they are grouped as `<name> cnn`.
- The results go to `data/explanations/<name>/cnn-bench/<id>/<dataset>/<variant>/`.
- No exclusions for the deep models yet, except the smoke limits above.

## Mode datasets

`make_mode_datasets.py` makes the datasets of smoke, short and long from
`data/datasets/<name>/<name>_s100_scaled.csv` (full uses that one directly). Sample numbers
are 1-based, as in the datasets and in spexplain's `sample [i/N]` output.

| benchmark | smoke | short (8) = first 8 of long | long: the 12 appended (samples 9–20) |
|---|---|---|---|
| heart_attack | 21, 55, 63, 84, 90, 98 | 14, 21, 55, 62, 63, 84, 90, 98 | 2, 4, 23, 25, 27, 36, 42, 48, 49, 50, 92, 99 |
| obesity | 77 | 2, 11, 24, 42, 45, 52, 81, 84 | 1, 3, 9, 29, 43, 51, 73, 74, 77, 79, 87, 88 |
| loan | 8, 70 | 8, 11, 24, 37, 38, 70, 81, 90 | 14, 19, 33, 43, 59, 65, 66, 71, 82, 87, 93, 95 |
| mnist | 1 | 1, 7, 11, 35, 41, 46, 84, 87 | 24, 26, 28, 33, 50, 51, 55, 59, 67, 78, 79, 90 |
| cifar | – (skipped) | 21, 30, 33, 55, 65, 84, 94, 97 | 3, 31, 34, 39, 41, 49, 51, 67, 68, 73, 95, 99 |
| gtsrb | – (skipped) | 21, 29, 44, 45, 49, 51, 59, 60 | 8, 13, 25, 31, 68, 71, 75, 78, 83, 84, 89, 93 |

The long dataset starts with the 8 samples of short, in the same order (so samples 1–8 of the
long results are those of short), followed by 12 more random ones, sorted. They are drawn with a
separate seed (`LONG_EXTRA_SEED`), so adding them changes neither the smoke nor the short dataset.

- **Important samples**, in every mode dataset: heart_attack 98, 63, 84, 55, 21, 90;
  loan 8, 70; mnist 1 and 7 (smoke: only 1). The other samples are random, with a fixed seed
  (`SEED` and `LONG_EXTRA_SEED` in the script), so re-running the script gives the same datasets.
- **Map** from the samples of the mode datasets to the original ones:
  `data/datasets/mode_sample_maps.csv`, columns `benchmark,mode,index,original_index`
  (full is included as the identity). Use it to relate e.g. sample 3 of the long results
  of heart_attack to its original sample 55.
- **Neuron activations**: the hybrid variant reads `data/neuron_activations/<name>/<model>.txt`,
  whose blocks are numbered by the samples of the dataset. With a smaller dataset, block 1
  would belong to the wrong sample, and spexplain would not notice. So the script also writes
  renumbered copies to `data/neuron_activations/<name>/s100_scaled_<mode>/<model>.txt`, which
  `run1.sh` uses for that dataset. They keep all blocks of each sample as in the original files
  (some list a sample twice, some lack samples), so hybrid behaves as on the full dataset.
  `run` stops if one of them is missing.
  The CNN models have their copies in
  `data/neuron_activations/<name>/cnn-bench/s100_scaled_<mode>/<id>.txt`, used by
  `data/scripts/cnn/run1.sh`. For them, `run` only prints
  `WARNING: neuron activations not found: <file>`.
- After changing the important samples or adding models, run the script again:
  `python3 make_mode_datasets.py`. The results of the earlier mode datasets are then outdated.

Jobs per benchmark and width:

| benchmark | smoke | short | long | full |
|---|---|---|---|---|
| heart_attack 50x | 19 | 137 | 137 | 142 |
| obesity 50x | 15 | 115 | 115 | 120 |
| loan 50x | 19 | 135 | 135 | 140 |
| mnist 50x | 19 | 175 | 175 | 200 |
| cifar 50x | – | 150 | 150 | 180 |
| gtsrb 50x | – | 150 | 150 | 180 |
| heart_attack 200x | – | 24 | 24 | 120 |
| obesity 200x | – | 24 | 24 | 120 |
| loan 200x | – | 24 | 24 | 120 |
| mnist 200x | – | 29 | 29 | 180 |
| cifar 200x | – | 23 | 23 | 180 |
| gtsrb 200x | – | 23 | 23 | 180 |
| mnist cnn | 18 | 48 | 96 | 96 |
| cifar cnn | – | 48 | 96 | 96 |
| gtsrb cnn | – | 48 | 96 | 96 |

## Runtime

All numbers assume a machine with **16 CPUs** and the default `CPU_PERCENTAGE=80`, i.e.
**13 jobs at once**, and the final model set, without the CNN jobs. For another machine, run e.g.
`CPUS=<n> ./runme.sh run long -n`.

**Max** = every explanation hits its timeout. This is a hard upper bound, apart from the
small model-loading overhead per job. The wall time simulates the shared queue (the
longest jobs first, each to the next free slot). These numbers are exact and printed by `-n`.

**Expected** is an estimate. The only timings available are for small heart_attack/loan
models, where `guide-all`, `fix-all` and `hybrid` take well below a second up to a few seconds
per sample, while `naive` takes tens of seconds and often reaches the timeout already for
50x3. There are no timings yet for mnist, cifar, gtsrb or any 200x model. "Naive-bound"
below assumes that every `naive` explanation times out and the other variants are fast,
which is the most likely bad case. Calibrate it with the joblog of the smoke run.

| | `smoke` | `short` | `long` | `full` |
|---|---|---|---|---|
| worst-case CPU time | 4h 22m | 729h | 2,839h | 27,133h |
| longest single job (max) | 6m | 2h 40m (ucore naive) | 6h 40m (ucore naive) | 33h 20m (ucore naive) |
| **max wall time** | **~20m** | **56h 13m (~2.3 days)** | **219h (~9.1 days)** | **2,090h (~87 days)** |
| naive-bound wall time | – (no naive) | 7h 20m | 29h 20m | 513h 20m (~21 days) |
| **expected** | **~10–15m** | **~4–8h** | **~15–30h** | **~3 weeks** |

Notes:
- `smoke` skips cifar and gtsrb, runs only the models up to 50x6, no `naive` and no ucore
  jobs; its 72 jobs explain 1–6 samples each (heart_attack 6, loan 2, mnist 1, obesity 1).
  Even if every explanation times out, it finishes in about 20m (CPU time / 13 slots).
- `short` and `long`: with 13 slots the queue is always full, so their wall time is about
  the CPU time divided by 13. The exclusions remove the deep `naive` jobs, so the remaining
  `naive` jobs (149, up to 365h CPU time in long) are no longer most of the worst case:
  that is now the other variants on the large image models (798 jobs, up to 2,328h in long).
  Every job of long explains 20 samples instead of the 8 of short, so with the same timeouts
  all its times would be 2.5× those of short.
  Whether those time out is unknown until the smoke run.
- `full` is not practical on 16 CPUs: even the naive-bound case takes about 3 weeks.
  It needs a cluster node (e.g. 128 jobs at once: naive-bound ~2.5 days) or the
  reductions below. Under Slurm, resubmit the same command with `--resume` after the 24h
  limit; a job killed by the limit starts from scratch on resume.
- With 80%, 13 jobs run on 16 CPUs. If these are 8 physical cores with hyperthreading,
  some jobs share a core and run slower than one job per core, which also means more timeouts.

### Effect of the exclusions, 8 samples and the short timeouts

Compared with the earlier `long` (then named `short`: the first 10 samples, no exclusions), same 16 CPUs.
"long, 8 samples" here is `long` when it had the 8 samples of `short`, before it was enlarged to 20
(multiply its times by 2.5):

| | long before (10 samples) | long, 8 samples | short |
|---|---|---|---|
| jobs | 1,080 | 947 | 947 |
| explanations | 10,800 | 7,576 | 7,576 |
| longest job | 2h 20m | 1h 52m | 1h 20m |
| naive-bound | 31h | 11h 44m | 7h 20m |
| max (everything times out) | 122h | 83h | 53h |
| expected | ~1–1.5 days | ~6–12h | ~4–8h |

The exclusions remove the deep `naive` jobs, which did most of the work (`naive` CPU time
in long: 396h → 146h). Eight samples instead of ten cut another 20% from everything,
and the lower timeouts of `short` cut about 35% more.

### Shared queue vs. one call per benchmark and width

Before, `run-experiments.sh` was called once per benchmark and width, one call after another,
and each call waited for its slowest job. Now all jobs share one queue. The time per
explanation is the same; only the idle cores at the end of each call are gone.
16 CPUs, 80% → 13 jobs at once:

| | one call per benchmark/width (before) | one shared queue (now) | saved |
|---|---|---|---|
| `short`, worst case | 55h 45m | 53h 20m | ~4% |
| `short`, only naive times out | 12h 36m | 7h 20m | ~42% |
| `long` (8 samples), worst case | 86h 48m | 82h 56m | ~4% |
| `long` (8 samples), only naive times out | 19h 36m | 11h 44m | ~40% |
| `full`, worst case | 2,086h 40m | 2,036h 40m | ~2% |
| `full`, only naive times out | 603h 20m | 513h 20m | ~15% |

The shared queue helps when there are more free slots than jobs at the end of a call, which
is rare with 13 slots, except when only a few long `naive` jobs are left. On a 128-job cluster node, it saves 40–80% for `long`.

## Examples

```bash
# check what would run on a 16-CPU machine
CPUS=16 ./runme.sh run long -n

# smoke test locally (needs bash >= 4, e.g. Homebrew's)
/opt/homebrew/bin/bash ./runme.sh run smoke

# full run on Eiger; resubmit with --resume after the 24h limit
WRAPPER=~/srun_commands/sbatch.sh ./runme.sh run full
WRAPPER=~/srun_commands/sbatch.sh ./runme.sh run full --resume

# only some jobs, e.g. rerun mnist 200x hybrid
FILTER='^mnist/200x[0-9]+/hybrid/' ./runme.sh run full
```

## Unsat cores of interpolants

In every mode but `smoke`, `run` also has the jobs `heart_attack/50x1/{naive,fix-all}/ucore_min_itp_aweak_bstrong`:
interpolants reduced by minimal unsat cores, strategy `itp aweak, bstrong; ucore min` (the
interpolant is computed in the same run, then reduced), i.e. ITP+ucore min of the ucore table.

- They are ordinary jobs of the queue: one `spexplain` process per variant, the samples
  explained sequentially, on the mode's dataset; `-n`, `--resume` and `FILTER` apply to them
  (e.g. `FILTER=ucore` runs only them).
- They run with OpenSMT: `run_job` adds `-V opensmt` (to `OPTIONS` of `run1.sh`).
- Timeout per explanation: 20m in every mode that runs them (as in the ucore table).
- The results go next to the others, `.../heart_attack/50x1/<dataset>/<variant>/ucore_min_itp_aweak_bstrong.*`;
  in `full` they overwrite those of the ucore table.
- In every mode that runs them, the `naive` job is the longest of all (short 8 × 20m = 2h 40m,
  long 20 × 20m = 6h 40m). In `full`: up to 100 × 20m = 33h 20m, more than the
  24h Slurm limit (in the earlier run, 46 of 100 samples hit the limit: ~16–20h). On resume,
  a killed job starts from scratch.
- What and where: `UCORE_EXPERIMENTS`, `ucore_experiments` and `mode_ucore_timeout_per` in `modes.sh`.
- `collect-stats` does not collect them by default (`STATS_FILTER`). When included, e.g.
  `STATS_FILTER=.`, it collects them with the `ucore_itp` experiments collection (`UCORE_EXPERIMENTS_SPEC`),
  as the last rows of the heart_attack 50x1 tables, so the header keeps the timeout of the
  itp experiments. `plot-performance` leaves them out of the plots (they are only in the CSV).

## Collecting the stats

```bash
./runme.sh collect-stats <mode> [-n]
```

`collect-stats` runs `collect_stats.sh` on the results of the given mode and writes one file
per benchmark, with both widths, to `./stats/<mode>/<name>.txt`, e.g. `./stats/long/cifar.txt`
(`STATS_DIR` changes `./stats`). The CNN models go to their own file `./stats/<mode>/<name>_cnn.txt`,
with headers like `mnist NAB1 hybrid`; `plot-performance` plots them with `plot_cnn_stats.py`.

- It uses the same job list as `run`, so it only collects what the mode runs: the right
  results folder (`s100_scaled_<mode>/` or `s100_scaled/` for full), only the
  experiments of the mode, and none of the excluded jobs. `FILTER` works the same way.
- Of those, it collects only `naive`, `guide-all`, `fix-all` and `hybrid`, and only the itp
  experiments (`itp_*`), i.e. what the plots show. `STATS_FILTER` (a regex over the job ids,
  default in `modes.sh`) changes this, e.g. `STATS_FILTER=.` collects everything.
  `fix-all-no-vars` is left out because `collect_stats.sh` rejects it: its consistency check
  compares the `#terms` of `.stats.txt` with the term size of `.phi.txt`, and with
  `--allow-neuron-vars-in-explanations false` the written explanations have more terms than
  the engine reports (e.g. 147.0 vs 230.5), so every such experiment fails with
  "encountered inconsistency: stats.termSize !~ termSize(phi)".
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
`./collect_stats.sh explanations/heart_attack/50x1/s100_scaled_long/hybrid/ plain_itp`.

## Plotting

```bash
./runme.sh plot-performance <mode> [-n]
```

`plot-performance` runs `plot_fc_stats.py` on the stats of `collect-stats` (run that first)
and writes the PAR-2 plots to `data/output/<mode>/` (next to the interpretation figures):
`fig-5-performance-plots_PAR-2_all_tasks_<width>.pdf` (all six benchmarks), for the widths 50 and 200.
The parsed tables go to the same folder as `fig-5-performance-plots_<name>.csv`. The stats themselves stay in
`./stats/<mode>/`.

It then runs `plot_cnn_stats.py` on the CNN stats (`<name>_cnn.txt`): one plot per CNN model
family, `fig-9_CNN_performance-plots_PAR-2_<family>.pdf` (e.g. `narrow-avg-pool` for NAB,
`narrow-stride2` for NS), with mnist, cifar and gtsrb side by side and the PAR-2 over the
#ReLUs of the models (from `data/models/<name>/cnn-bench/manifest.csv`). Its tables are
`fig-9_CNN_performance-plots_<name>_cnn.csv`.

- It uses the Python of `venv/` in the repository, created with
  `python3 -m venv venv && venv/bin/pip install -r requirements.txt`; `PYTHON` overrides it.
- The timeout lines and the y-limits use the timeout per sample of the stats, so they fit
  every mode. Missing results are not drawn: a line simply has no point at that model.
- Benchmarks without stats keep their place in the grid but stay empty, and widths (CNN:
  families) without results are skipped (smoke: only the 50x plots, without cifar and gtsrb).
  The y-limit is 2 × timeout + 5 s (tabular 50x: 30 s, 200x: + 2 s).
- With LaTeX installed, text is rendered with it; without, with Matplotlib's mathtext.
- What to plot (plotted depths, the itp algorithms of 200x, colors, fonts) is set at the top of
  `plot_fc_stats.py` and in `ploting_utils.py`; for the CNN models (families, itp algorithms)
  at the top of `plot_cnn_stats.py`.
- With the results of `run-verix-plus` of the mode, it also compares our results with VeriX+.
  First it writes the run times of VeriX+ into `data/output/<mode>/`: `fig-8_comparison-plots_<name>_verix.csv`
  (50x) and `fig-8_comparison-plots_<name>200_verix.csv` (200x), with the columns `Algorithm,epsilon,layers,PAR2`
  as `produce_csv.py` of the original experiments: the average PAR-2 of the samples per epsilon (e.g. `5%`)
  and model depth (`layers`, from 1), with the mode's timeout per explanation. The time of a sample is its
  `time.txt` of `verix1.py`; one without it (a timeout or a failure), or with at least the timeout, counts as
  twice the timeout. The algorithm is `IR`, the random traversal order of `run-verix-plus`; the models without
  its results are left out.
  Then `plot_compare_with_verix_plus.py <stats_dir> [<output_dir>]` reads our stats as `plot_fc_stats.py` does
  (the same `read_stats` of `ploting_utils.py`, also the timeouts) and the `IR` rows of these CSVs, and writes
  `fig-8_comparison-plots_PAR-2_ours_vs_verix_plus_all_tasks_<width>.pdf` next to them.

## Interpretation figures

```bash
./runme.sh interpret <mode> [-n]
```

`interpret` runs `runme_norm.sh <mode>`, `runme_diam.sh <mode>`, `runme_methodes.sh <mode>` and
`runme_ucore.sh <mode>`, in this order, on the mode's explanations (of `run`, and of `run-verix-plus`).
The figures go to `data/output/<mode>/`.

- `runme_norm.sh` (Fig. 7): `interpret ... 'norm min, inv' --visualize 'pixel center'` of mnist
  50x1, sample 1, with the variants `guide-all`, `fix-all-no-vars`, `verix/eps5` and `verix/eps100`.
  Smoke computes only `'norm min'`, without `inv`.
- `runme_diam.sh` (Fig. 11): `interpret ... 'diam vars <x> <y>'` of the 50x1 `guide-all`
  `itp_aweak_bstrong` explanations, as an image and as `mathexpr` (PDF): heart_attack sample 21
  (s20 in the paper) and loan sample 90 (s89 in the paper), four pairs of variables each.
- `runme_methodes.sh` (Fig. 6): `interpret ... 'slice vars <x> <y>' --visualize fspace` of the
  `naive`, `guide-all` and `fix-all-no-vars` `itp_aweak_bstrong` explanations together with `verix/eps5`
  (of `run-verix-plus`), for 50x1 and 50x4: heart_attack samples 98 and 63 (s97, s62 in the paper)
  and loan samples 8 and 70 (s7, s69 in the paper). A missing variant is left out of the figure; smoke has
  no loan `naive`, so its loan figures show only `guide-all` and `fix-all-no-vars` (and `verix/eps5`).
- `runme_ucore.sh` (Fig. 10): `interpret ... 'slice vars <x> <y>' --visualize fspace` of the
  heart_attack 50x1 `ucore_min_itp_aweak_bstrong` and `itp_aweak_bstrong` explanations together
  (the ucore min region underneath), one figure per variant, `naive` and `fix-all`: samples 84
  (s83 in the paper, vars x8 x11) and 55 (s54 in the paper, vars x5 x7). Both explanations exist
  only after `run` of the mode (the ucore jobs); a missing one is left out of the figure.
- The samples are the indices in the original s100 dataset. `mode_sample` (in `modes.sh`)
  turns them into the indices in the mode's dataset via `data/datasets/mode_sample_maps.csv`.
  Sample 90 of loan is not in the smoke dataset, so smoke has no loan figures.
- A missing explanations file or sample is skipped with a message (e.g. verix without
  `run-verix-plus` of the mode). A failed `spexplain interpret` makes the script, and then `interpret`
  or `run`, exit with 1.
- The explanations must have been computed on the current mode dataset. After
  `make_mode_datasets.py` changes it, run the mode again, or `interpret` fails with
  "The number of explanations ... does not match the size of the dataset".
- They use the Python of `venv/` in the repository (`PYTHON` overrides it), as `spexplain
  interpret` needs matplotlib.
- The scripts can also be run on their own: `./runme_norm.sh <mode>`, `./runme_diam.sh <mode>`,
  `./runme_methodes.sh <mode>` (default mode: smoke).

## VeriX-plus

```bash
./runme.sh run-verix-plus <mode> [--resume] [-n]
```

`run-verix-plus` computes the explanations of VeriX-plus (https://github.com/minwu-cs/VeriX-plus),
as in the original experiments, on the same models and mode datasets as `run` with the same timeouts
per explanation, and with the perturbations epsilon 1, 0.05 and 0.005.

- The configuration of the original experiments, as in the image `docker/tacas27`:
  - VeriX-plus at commit `5efb636`, with the patch of the original experiments for binary classification
    (`docker/tacas27/verix-tabular.patch`): the dataset `tabular`, perturbed per feature as mnist, and for
    networks with a single output, whether the output can change its sign. The patch takes the predicted label
    as the argmax of the single output, i.e. always 0, so it always checks whether the output can be >= 0:
    the samples with a positive output are never robust and their explanations fix all features.
  - Marabou at commit `1c2f478`, built without Gurobi. VeriX-plus solves the queries of Marabou as MILP,
    which without Gurobi are `sat` unless refuted by the preprocessing of Marabou (bound tightening).
    The explanations are thus correct, as a feature is perturbed only if refuted, but not minimal, and fast.
    Gurobi was not used.
  - The random traversal order of VeriX-plus (with its seed 0), the binary search and the confidence ranking.
    `VERIX_ORDER` sets another order (e.g. `bounds`).
- The reference explanations of the original experiments (mnist 50x1 and 50x4, epsilon 1 and 0.05, samples 1-10)
  are reproduced up to a few features per explanation, with similar run times: the original setup could not be
  recovered in all details.
- All benchmarks (`VERIX_NAMES` in `modes.sh`), only the fully connected models (not the CNN models).
  The tabular samples are shaped `(<features>, 1, 1)`; their explanations are not plotted by VeriX-plus,
  whose plot needs images.
  The images of gtsrb and cifar have three channels: VeriX-plus takes the channels of a pixel together, whereas
  the datasets and the networks have them channel after channel, so `verix1.py` reorders the inputs of the
  networks that VeriX-plus reads, and its pixels are mapped back to the features.
- The jobs are model × epsilon, e.g. `mnist/50x1/verix/eps100`, in the same kind of GNU parallel queue
  as `run` (also `-n`, `--resume` and `FILTER`). Their job list and joblog are in
  `data/logs/runme/<mode>/verix/`.
- One job is `verix/verix1.sh`: it runs `verix/verix1.py` on each sample with the timeout per explanation,
  which makes the networks for VeriX-plus from the `.nnet` (with `verix/nnet_to_onnx.py`, not from an
  existing `.onnx`, which may be another conversion) into `eps<P>/models/`,
  and writes the explanations with `verix/to_phi.py` to
  `data/explanations/<name>/<width>x<depth>/<dataset>/verix/eps{100,5,0.5}.phi.txt`, where
  `runme_norm.sh` and `runme_methodes.sh` read them. The results of VeriX-plus of each sample (`sat.txt`,
  `unsat.txt`, `time.txt` and its log) are in the directory `eps<P>/` next to them.
- An explanation fixes the features of the explanation of VeriX-plus (`sat.txt`, pixels with all their
  channels) to the sample, and perturbs the others (`unsat.txt`) by epsilon, within [0,1]. A sample robust already with all features
  perturbed perturbs all of them. A timeout or a failure is `<null>`; a failure makes the job fail.
- The timeout kills VeriX-plus (`timeout --signal=KILL`): Marabou installs its handler of SIGTERM in the process
  with the first query, the queries as MILP never check it, and the handler refers to the finished queries,
  so SIGTERM does not stop VeriX-plus, or crashes it.
- VeriX-plus runs with one thread (`OPENBLAS_NUM_THREADS=1` etc.), as each job of the queue: the threads of
  OpenBLAS (of Marabou) do not speed it up, but the parallel jobs slow each other down with them
  (measured: about 1.8x slower with 6 jobs at once); the explanations are the same.
- With epsilon 0.005, many samples are robust already with all features perturbed (refuted by the
  preprocessing of Marabou), and their explanations are quick; the others run the whole search.
  The run times (`plot-performance`) count the robust samples with their time and a timeout as twice
  the timeout per explanation (PAR-2).
- It fails unless VeriX-plus can run as in the original experiments: installed (`VERIX_DIR`, default
  `~/VeriX-plus`, as in the Docker image), with Marabou without Gurobi.

## Files

- `runme.sh`: the actions, the job list and the GNU parallel queue.
- `modes.sh`: what each mode runs: benchmarks, widths, variants, experiments, exclusions,
  datasets and timeouts.
- `make_mode_datasets.py`: the mode datasets, the sample maps and their neuron activations
  (also of the CNN models).
- `../run1.sh`: one FC job. It no longer passes `--quiet` by default, because `--quiet` segfaults
  in some builds of the engine (`QUIET=1` turns it back on). So `.out.txt` holds a short
  progress log; the explanations and stats are unchanged.
- `../cnn/run1.sh`: one CNN job, with the engine of `make onnx`.
- `plot_fc_stats.py`, `plot_cnn_stats.py`, `ploting_utils.py`: the plots of `plot-performance`.
- `plot_compare_with_verix_plus.py`: the comparison with VeriX+ (not run yet by `plot-performance`).
- `runme_norm.sh`, `runme_diam.sh`, `runme_methodes.sh`, `runme_ucore.sh`: the interpretation figures of `interpret`.
- `verix/`: the jobs of `run-verix-plus`.


## Run Times on My Laptop
- CPU: Apple M1 Pro, 10 cores (8 performance, 2 efficiency)
- Smoke: 27 minutes, 30 seconds
````