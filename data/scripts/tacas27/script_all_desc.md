````# `runme.sh` — all TACAS'27 benchmarks in one run

```
./runme.sh <action> <mode> [--resume] [-n] [-h]
```

Actions: `run`, `collect-stats`, `plot-performance`, `interpret` (more to come: `run-verix-plus`).
Modes: `smoke`, `smokeLong`, `quick`, `short`, `full`. Both are required.

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
  `s100_scaled_smoke`, `s100_scaled_quick`, `s100_scaled_short` or `s100_scaled` (full), so
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

| | `smoke` | `quick` | `short` | `full` |
|---|---|---|---|---|
| benchmarks | all but cifar, gtsrb | all | all | all |
| widths | 50x, up to 50x6 | 50x, 200x | 50x, 200x | 50x, 200x |
| variants | guide-all, fix-all, hybrid, fix-all-no-vars | all 5, minus the exclusions below | all 5, minus the exclusions below | all 5, minus the fix-all-no-vars exclusions |
| experiments | itp_aweak_bstrong | 50x: all 5; 200x: itp_aweak_bstrong | 50x: all 5; 200x: itp_aweak_bstrong | all 5 |
| dataset | `<name>_s100_scaled_smoke.csv` | `<name>_s100_scaled_quick.csv` | `<name>_s100_scaled_short.csv` | `<name>_s100_scaled.csv` |
| samples | the important ones (1–6) | 8 (the first 8 of short) | 20 (those of quick, then 12 more) | all 100 |
| timeout per explanation | 1m tabular, 5m mnist | 200s tabular, MN 300s, CIF 400s, GTS 600s | HA/OB/LO 5m, MN/CIF 10m, GTS 14m | HA/OB/LO 5m, MN/CIF 10m, GTS 14m |
| on a failed job | stop | continue | continue | continue |
| results in | `s100_scaled_smoke/<variant>/` | `s100_scaled_quick/<variant>/` | `s100_scaled_short/<variant>/` | `s100_scaled/<variant>/` |
| **jobs** (FC + CNN) | **96** (72 + 24) | **1,153** (1,009 + 144) | **1,153** (1,009 + 144) | **2,006** (1,862 + 144) |
| **explanations** (FC + CNN) | **210** (186 + 24) | **9,224** (8,072 + 1,152) | **9,224** (8,072 + 1,152) | **200,600** (186,200 + 14,400) |

`fix-all-no-vars` runs in all modes, but only for the 50x1–50x4 models of heart_attack, loan and mnist.
`smoke` runs only the models up to 50x6, no `naive` and no unsat-core jobs.

**`smokeLong`** is a smoke test for us and will be removed later. It is `smoke` plus `naive` for
the heart_attack 50x1–50x4 models and the unsat-core jobs (see "Unsat cores of interpolants"):
78 FC jobs and 222 explanations, plus the 24 CNN jobs of `smoke`. It uses the smoke dataset, sample map and timeouts, and so writes into
the same results folder `s100_scaled_smoke/` (its jobs that `smoke` also has overwrite those
results). Its job log, stats and figures go under its own name (`logs/runme/smokeLong/`,
`stats/smokeLong/`, `output/smokeLong/`). Like `smoke`, it stops at the first failed job.
In `modes.sh`, `dataset_mode` maps it to `smoke` and `is_smoke_mode` covers both.

`smoke`, `short` and `quick` also leave out the deep models where a variant mostly times out
(for `smoke` these change nothing beyond that limit, as it runs no 200x jobs):

| variant | benchmarks | left out |
|---|---|---|
| naive | heart_attack, obesity, loan | 50x8 (and deeper) |
| naive | mnist | 50x6 and deeper, 200x3 and deeper |
| naive | cifar, gtsrb | 50x4 and deeper, 200x3 and deeper |
| guide-all (prefer-all) | cifar, gtsrb | 200x4 and deeper |

`quick` differs from `short` in the lower timeouts and in its samples: the first 8 of short
(the same jobs, separate results folders).

## CNN (ONNX) models

Every mode also runs the CNN-bench models of `data/scripts/cnn/tacas27/{MN,CIF,GTS}.sh`, as jobs of
the same GNU parallel queue (`CNN_*`, `set_cnn_mode` and `is_cnn_excluded` in `modes.sh`):

| | `smoke`, `smokeLong` | `quick`, `short`, `full` |
|---|---|---|
| benchmarks | mnist | mnist, cifar, gtsrb |
| models (`data/models/<name>/cnn-bench/`) | NS1–NS4, NAB1–NAB4 | NS1–NS6, NAB1–NAB6 |
| variants (`data/scripts/cnn/spec/variants/all`) | guide-all, fix-all, hybrid | naive, guide-all, fix-all, hybrid |
| experiments | itp_aweak_bstrong | itp_aweak_bstrong |
| **jobs** | **24** | **144** (48 per benchmark) |

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

`make_mode_datasets.py` makes the datasets of smoke, quick and short from
`data/datasets/<name>/<name>_s100_scaled.csv` (full uses that one directly). Sample numbers
are 1-based, as in the datasets and in spexplain's `sample [i/N]` output.

| benchmark | smoke | quick (8) = first 8 of short | short: the 12 appended (samples 9–20) |
|---|---|---|---|
| heart_attack | 21, 55, 63, 84, 90, 98 | 14, 21, 55, 62, 63, 84, 90, 98 | 2, 4, 23, 25, 27, 36, 42, 48, 49, 50, 92, 99 |
| obesity | 77 | 2, 11, 24, 42, 45, 52, 81, 84 | 1, 3, 9, 29, 43, 51, 73, 74, 77, 79, 87, 88 |
| loan | 8, 70 | 8, 11, 24, 37, 38, 70, 81, 90 | 14, 19, 33, 43, 59, 65, 66, 71, 82, 87, 93, 95 |
| mnist | 1 | 1, 7, 11, 35, 41, 46, 84, 87 | 24, 26, 28, 33, 50, 51, 55, 59, 67, 78, 79, 90 |
| cifar | – (skipped) | 21, 30, 33, 55, 65, 84, 94, 97 | 3, 31, 34, 39, 41, 49, 51, 67, 68, 73, 95, 99 |
| gtsrb | – (skipped) | 21, 29, 44, 45, 49, 51, 59, 60 | 8, 13, 25, 31, 68, 71, 75, 78, 83, 84, 89, 93 |

The short dataset starts with the 8 samples of quick, in the same order (so samples 1–8 of the
short results are those of quick), followed by 12 more random ones, sorted. They are drawn with a
separate seed (`SHORT_EXTRA_SEED`), so smoke and quick are the same as before short was enlarged.

- **Important samples**, in every mode dataset: heart_attack 98, 63, 84, 55, 21, 90;
  loan 8, 70; mnist 1 and 7 (smoke: only 1). The other samples are random, with a fixed seed
  (`SEED` and `SHORT_EXTRA_SEED` in the script), so re-running the script gives the same datasets.
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
  The CNN models have their copies in
  `data/neuron_activations/<name>/cnn-bench/s100_scaled_<mode>/<id>.txt`, used by
  `data/scripts/cnn/run1.sh`. For them, `run` only prints
  `WARNING: neuron activations not found: <file>`.
- After changing the important samples or adding models, run the script again:
  `python3 make_mode_datasets.py`. The results of the earlier mode datasets are then outdated.

Jobs per benchmark and width:

| benchmark | smoke | quick, short | full |
|---|---|---|---|
| heart_attack 50x | 19 (smokeLong: 25) | 137 | 142 |
| obesity 50x | 15 | 115 | 120 |
| loan 50x | 19 | 135 | 140 |
| mnist 50x | 19 | 175 | 200 |
| cifar 50x | – | 150 | 180 |
| gtsrb 50x | – | 150 | 180 |
| heart_attack 200x | – | 24 | 120 |
| obesity 200x | – | 24 | 120 |
| loan 200x | – | 24 | 120 |
| mnist 200x | – | 29 | 180 |
| cifar 200x | – | 23 | 180 |
| gtsrb 200x | – | 23 | 180 |
| mnist cnn | 24 | 48 | 48 |
| cifar cnn | – | 48 | 48 |
| gtsrb cnn | – | 48 | 48 |

## Runtime

All numbers assume a machine with **16 CPUs** and the default `CPU_PERCENTAGE=80`, i.e.
**13 jobs at once**, and the final model set, without the CNN jobs. For another machine, run e.g.
`CPUS=<n> ./runme.sh run short -n`.

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
| worst-case CPU time | 4h 22m | 729h | 2,839h | 27,133h |
| longest single job (max) | 6m | 2h 40m (ucore naive) | 6h 40m (ucore naive) | 33h 20m (ucore naive) |
| **max wall time** | **~20m** | **56h 13m (~2.3 days)** | **219h (~9.1 days)** | **2,090h (~87 days)** |
| naive-bound wall time | – (no naive) | 7h 20m | 29h 20m | 513h 20m (~21 days) |
| **expected** | **~10–15m** | **~4–8h** | **~15–30h** | **~3 weeks** |

Notes:
- `smoke` skips cifar and gtsrb, runs only the models up to 50x6, no `naive` and no ucore
  jobs; its 72 jobs explain 1–6 samples each (heart_attack 6, loan 2, mnist 1, obesity 1).
  Even if every explanation times out, it finishes in about 20m (CPU time / 13 slots).
- `smokeLong` (78 jobs): worst case 8h 46m CPU time and 2h wall time, all of it the `ucore`
  `naive` job (6 × 20m). The `naive` jobs likely time out from 50x3 on.
- `short` and `quick`: with 13 slots the queue is always full, so their wall time is about
  the CPU time divided by 13. The exclusions remove the deep `naive` jobs, so the remaining
  `naive` jobs (149, up to 365h CPU time in short) are no longer most of the worst case:
  that is now the other variants on the large image models (798 jobs, up to 2,328h in short).
  Every job of short explains 20 samples instead of the 8 of quick, so with the same timeouts
  all its times would be 2.5× those of quick.
  Whether those time out is unknown until the smoke run.
- `full` is not practical on 16 CPUs: even the naive-bound case takes about 3 weeks.
  It needs a cluster node (e.g. 128 jobs at once: naive-bound ~2.5 days) or the
  reductions below. Under Slurm, resubmit the same command with `--resume` after the 24h
  limit; a job killed by the limit starts from scratch on resume.
- With 80%, 13 jobs run on 16 CPUs. If these are 8 physical cores with hyperthreading,
  some jobs share a core and run slower than one job per core, which also means more timeouts.

### Effect of the exclusions, 8 samples and the quick timeouts

Compared with the earlier `short` (first 10 samples, no exclusions), same 16 CPUs.
"short now" here is `short` with 8 samples, before it was enlarged to 20 (multiply its times by 2.5):

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
| `short` (8 samples), worst case | 86h 48m | 82h 56m | ~4% |
| `short` (8 samples), only naive times out | 19h 36m | 11h 44m | ~40% |
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

## Explanations for interpretation

At the end of `run`, in every mode, after the GNU parallel queue has finished, `runme.sh` also
runs `commands_fix-all_no_neuron_vars.sh`. It computes explanations without inner-neuron
variables (`--allow-neuron-vars-in-explanations false`), suitable for visualization:

- loan and obesity, 50x1–50x4, variants `naive` (Direct), `guide-all` and `fix-all`;
- `itp_astrong_bstrong`, `itp_afactor_0.5_bstrong`, `itp_aweak_bstrong`;
- always on the `<name>_s100_scaled_quick.csv` datasets (8 samples), whatever the mode is,
  with a 10 min limit per explanation;
- the 72 runs go one after another (not through the queue), into
  `data/explanations_for_interpretation/<name>/50x<d>/s100_scaled_quick/<variant>/`,
  overwriting earlier results.

`-n` does not run it. If it fails, `run` reports it and exits with 1. The overall running time
printed by `run` includes it. It can also be run on its own:
`./commands_fix-all_no_neuron_vars.sh` (needs bash >= 4).

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
- In every mode that runs them, the `naive` job is the longest of all (smokeLong 6 × 20m = 2h, quick and short
  8 × 20m = 2h 40m). In `full`: up to 100 × 20m = 33h 20m, more than the
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
per benchmark, with both widths, to `./stats/<mode>/<name>.txt`, e.g. `./stats/short/cifar.txt`
(`STATS_DIR` changes `./stats`). The CNN models go to their own file `./stats/<mode>/<name>_cnn.txt`,
with headers like `mnist NAB1 hybrid`; `plot-performance` does not plot them yet.

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
`./collect_stats.sh explanations/heart_attack/50x1/s100_scaled_short/hybrid/ plain_itp`.

## Plotting

```bash
./runme.sh plot-performance <mode> [-n]
```

`plot-performance` runs `plot_fc_stats.py` on the stats of `collect-stats` (run that first)
and writes the PAR-2 plots to `data/output/<mode>/` (next to the interpretation figures):
`performance-plots_PAR-2_all_tasks_<width>.pdf` (all six benchmarks) and
`performance-plots_PAR-2_image_only_<width>.pdf`, for the widths 50 and 200. The parsed tables
go to the same folder as `performance-plots_<name>.csv`. The stats themselves stay in
`./stats/<mode>/`.

- It uses the Python of `venv/` in the repository, created with
  `python3 -m venv venv && venv/bin/pip install -r requirements.txt`; `PYTHON` overrides it.
- The timeout lines and the y-limits use the timeout per sample of the stats, so they fit
  every mode. Missing results are not drawn: a line simply has no point at that model.
- Benchmarks without stats keep their place in the grid but stay empty, and widths without
  results are skipped (smoke: only the 50x plots, without cifar and gtsrb).
- With LaTeX installed, text is rendered with it; without, with Matplotlib's mathtext.
- What to plot (plotted depths, the itp algorithms of 200x, colors, fonts) is set at the top of
  `plot_fc_stats.py` and in `ploting_utils.py`.
- Not run yet (commented out in `runme.sh`, until the VeriX+ results are available):
  `plot_compare_with_verix_plus.py <stats_dir> [<output_dir>]` compares our results with VeriX+.
  It reads our stats as `plot_fc_stats.py` does (the same `read_stats` of `ploting_utils.py`, also
  the timeouts), and the VeriX+ CSVs from `data/output/<mode>/`: `performance-plots_<name>_verix.csv`
  (50x) and `performance-plots_<name>200_verix.csv` (200x), columns `Algorithm,epsilon,layers,PAR2`.
  It writes `performance-plots_PAR-2_ours_vs_verix_plus_<layout>_<width>.pdf` next to them.

## Interpretation figures

```bash
./runme.sh interpret <mode> [-n]
```

`interpret` runs `runme_norm.sh <mode>`, `runme_diam.sh <mode>`, `runme_methodes.sh <mode>` and
`runme_ucore.sh <mode>`, in this order, on the mode's explanations. `run` also runs all four at the end, in every mode, after the GNU parallel queue
(not with `-n`). The figures go to `data/output/<mode>/`.

- `runme_norm.sh` (Fig. 7): `interpret ... 'norm min, inv' --visualize 'pixel center'` of mnist
  50x1, sample 1, with the variants `guide-all`, `fix-all-no-vars`, `verix/eps5` and `verix/eps100`.
- `runme_diam.sh` (Fig. 11): `interpret ... 'diam vars <x> <y>'` of the 50x1 `guide-all`
  `itp_aweak_bstrong` explanations, as an image and as `mathexpr` (PDF): heart_attack sample 21
  (s20 in the paper) and loan sample 90 (s89 in the paper), four pairs of variables each.
- `runme_methodes.sh` (Fig. 6): `interpret ... 'slice vars <x> <y>' --visualize fspace` of the
  `naive`, `guide-all` and `fix-all-no-vars` `itp_aweak_bstrong` explanations together, for
  50x1 and 50x4: heart_attack samples 98 and 63 (s97, s62 in the paper) and loan samples 8 and
  70 (s7, s69 in the paper). A missing variant is left out of the figure; smoke has no loan
  `naive`, so its loan figures show only `guide-all` and `fix-all-no-vars`.
- `runme_ucore.sh` (Fig. 10): `interpret ... 'slice vars <x> <y>' --visualize fspace` of the
  heart_attack 50x1 `ucore_min_itp_aweak_bstrong` and `itp_aweak_bstrong` explanations together
  (the ucore min region underneath), one figure per variant, `naive` and `fix-all`: samples 84
  (s83 in the paper, vars x8 x11) and 55 (s54 in the paper, vars x5 x7). Both explanations exist
  only after `run` of the mode (the ucore jobs); a missing one is left out of the figure.
- The samples are the indices in the original s100 dataset. `mode_sample` (in `modes.sh`)
  turns them into the indices in the mode's dataset via `data/datasets/mode_sample_maps.csv`.
  Sample 90 of loan is not in the smoke dataset, so smoke has no loan figures.
- A missing explanations file or sample is skipped with a message (e.g. verix until
  `run-verix-plus` exists). A failed `spexplain interpret` makes the script, and then `interpret`
  or `run`, exit with 1.
- The explanations must have been computed on the current mode dataset. After
  `make_mode_datasets.py` changes it, run the mode again, or `interpret` fails with
  "The number of explanations ... does not match the size of the dataset".
- They use the Python of `venv/` in the repository (`PYTHON` overrides it), as `spexplain
  interpret` needs matplotlib.
- The scripts can also be run on their own: `./runme_norm.sh <mode>`, `./runme_diam.sh <mode>`,
  `./runme_methodes.sh <mode>` (default mode: smoke).

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
- `plot_fc_stats.py`, `ploting_utils.py`: the plots of `plot-performance`.
- `plot_compare_with_verix_plus.py`: the comparison with VeriX+ (not run yet by `plot-performance`).
- `runme_norm.sh`, `runme_diam.sh`, `runme_methodes.sh`, `runme_ucore.sh`: the interpretation figures of `interpret`, also run at
  the end of `run`.
- `commands_fix-all_no_neuron_vars.sh`: the explanations for interpretation, run at the end of `run`.


## Run Times on My Laptop
- CPU: Apple M1 Pro, 10 cores (8 performance, 2 efficiency)
- Smoke: 27 minutes, 30 seconds
````