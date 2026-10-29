# Spexplain TACAS'27 artifact: run times, storage, test machine, modes

Raw data for the artifact README. Every number is marked as **measured** (from real runs)
or **estimated** (extrapolated). Date of the data: 2026-10-27; `long` and `full` updated on 2026-10-29
for the CNN families S and AB (estimated, not yet measured).
All experiments are started with `data/scripts/tacas27/runme.sh <action> <mode>`.

## 1. Summary (machine-readable)

```yaml
test_machine:
  model: MacBook Pro (Mac15,3)
  cpu: Apple M3
  architecture: arm64
  cores: 8 (4 performance + 4 efficiency), no hyperthreading
  ram_gb: 16
  disk_gb: 926
  os: macOS 26.6.2
  tools: GNU bash 5.3.9, GNU parallel 20260922, Python 3.11.11 (venv), cmake 3.28.3, Apple clang 21.0.0
  parallel_jobs: 6          # CPUS=8, CPU_PERCENTAGE=80 (default) -> 6 jobs at once

modes:                      # wall time on the test machine (6 jobs at once)
  smoke:
    jobs: 90
    explanations: 204
    wall_time: "~9 min"                 # measured
    worst_case_wall_time: "59 min"      # every explanation hits its timeout
    storage_explanations: "0.5 GB"      # measured
  short:
    jobs: 1153
    explanations: 9224
    wall_time: "40-52 h (~2 days)"      # estimated from measured runs of 1017 of the 1153 jobs
    worst_case_wall_time: "145 h"
    storage_explanations: "~105-110 GB" # 101 GB measured without most CNN jobs
  long:
    jobs: 1297
    explanations: 25940
    wall_time: "215-283 h (~9-12 days)" # estimated
    worst_case_wall_time: "655 h"
    storage_explanations: "~270-315 GB" # estimated (2.5x short, plus the S and AB CNN models)
  full:
    jobs: 2150
    explanations: 215000
    wall_time: "not practical on a laptop (weeks)"  # estimated
    storage_explanations: "~1.4-1.5 TB" # estimated (12.5x short, plus the S and AB CNN models)
```

## 2. Modes

| | `smoke` | `short` | `long` | `full` |
|---|---|---|---|---|
| purpose | check that everything runs (AE smoke test) | representative subset of the paper's experiments | larger subset | all experiments of the paper |
| benchmarks | heart_attack, obesity, loan, mnist (no cifar, gtsrb) | all 6: heart_attack, obesity, loan, mnist, cifar, gtsrb | all 6 | all 6 |
| FC models (`.nnet`) | 50x1-50x6 only | 50x and 200x | 50x and 200x | 50x and 200x |
| CNN models (`.onnx`) | mnist NS1-NS3, NAB1-NAB3 | mnist, cifar, gtsrb NS1-NS6, NAB1-NAB6 | mnist, cifar, gtsrb NS1-NS6, NAB1-NAB6, S1-S6, AB1-AB6 | same as long |
| variants | guide-all, fix-all, hybrid, fix-all-no-vars (no naive) | naive, guide-all, fix-all, hybrid, fix-all-no-vars | same as short | same as short |
| interpolation experiments | itp_aweak_bstrong | 50x: all 5; 200x: itp_aweak_bstrong | 50x: all 5; 200x: itp_aweak_bstrong | all 5 |
| samples per dataset | 1-6 (the "important" samples) | 8 | 20 (the 8 of short + 12 more) | 100 |
| timeout per explanation | 1 min tabular, 5 min mnist | 200 s tabular, 300 s mnist, 400 s cifar, 600 s gtsrb | 5 min tabular, 10 min mnist/cifar, 14 min gtsrb | same as long |
| unsat-core jobs (heart_attack 50x1, timeout 20 min) | no | yes | yes | yes |
| on a failed job | stop at the first failure | continue | continue | continue |
| jobs (FC + CNN) | 90 (72 + 18) | 1,153 (1,009 + 144) | 1,297 (1,009 + 288) | 2,150 (1,862 + 288) |
| explanations (FC + CNN) | 204 (186 + 18) | 9,224 (8,072 + 1,152) | 25,940 (20,180 + 5,760) | 215,000 (186,200 + 28,800) |
| results directory (under `data/explanations/<benchmark>/<model>/`) | `s100_scaled_smoke/` | `s100_scaled_short/` | `s100_scaled_long/` | `s100_scaled/` |

- Tabular benchmarks: heart_attack, obesity, loan. Image benchmarks: mnist, cifar, gtsrb.
- CNN families: NS (narrow, stride 2), NAB (narrow, average pooling), S (stride 2), AB (average
  pooling). S and AB have about twice the ReLUs of NS and NAB and run only in `long` and `full`.
- `smoke`, `short` and `long` leave out the deep models where a variant mostly times out
  (e.g. naive on 50x8 and deeper for tabular, on 50x4 and deeper for cifar/gtsrb).
- All jobs share one GNU parallel queue; within a job the samples run sequentially.
- `run` also makes the interpretation figures at the end. After `run`, use the actions
  `collect-stats` and then `plot-performance` to make the performance plots in `data/output/<mode>/`.
- Full details: `data/scripts/tacas27/script_all_desc.md`.

### Commands

```bash
cd data/scripts/tacas27
./runme.sh run smoke -n                  # dry run: list the jobs only
./runme.sh run smoke                     # run
./runme.sh run short --resume            # continue an interrupted run, skipping finished jobs
CPUS=8 CPU_PERCENTAGE=80 ./runme.sh run short   # number of CPUs / share of them to use
./runme.sh collect-stats short && ./runme.sh plot-performance short
```

- Number of parallel jobs = round(`CPUS` x `CPU_PERCENTAGE` / 100). Default `CPU_PERCENTAGE` is 80;
  default `CPUS` is all CPU threads.
- Logs: `data/logs/runme/<mode>/jobs.txt` (job list), `data/logs/runme/<mode>/joblog.txt`
  (one line per finished job: run time, exit value).
- Requires bash >= 4 and GNU parallel. The CNN jobs also need the ONNX engine (`make onnx`).

## 3. Run times

### 3.1 Measured on the test machine (6 jobs at once)

| mode | measured wall time | coverage |
|---|---|---|
| smoke | ~9 min | complete run |
| short | 1,017 of the 1,153 jobs ran; 161 job-hours of CPU time | all FC jobs and gtsrb CNN NS1-NS3; the remaining 136 CNN jobs (mnist, cifar, gtsrb NS4-NS6/NAB) were not yet measured |

"Job-hours" = sum of the run times of all jobs. Wall time ~= job-hours / parallel jobs.

### 3.2 Estimated wall time

Method: measured run time per job where available; the unmeasured CNN jobs and the jobs that
failed are estimated between "like the measured gtsrb CNN jobs" (low) and "every explanation times
out" (high). `long` = 2.5x the samples of short, with the longer timeouts applied in proportion to
how often a job hit its timeout in short. The queue is simulated with the longest jobs first.
The S and AB CNN models of `long` (144 jobs, 544 job-hours if every explanation times out) have no
measurements; they are estimated between 85% of their worst case (the share the measured gtsrb NS
jobs of short used) and 100%, since they are about twice the size of NS and NAB.

| mode | job-hours | test machine (8 cores, 6 jobs) | server, 60 cores (48 jobs) | server, 128 cores (102 jobs) |
|---|---|---|---|---|
| smoke | ~1 | ~9 min (measured) | ~5 min | ~5 min |
| short | 240-340 | 40-52 h (~2 days) | 5-7 h | 2.5-3.5 h |
| long | 1,290-1,795 | 215-283 h (~9-12 days) | 28-43 h | 14-21 h |
| full | - | weeks (not practical) | several days | ~2.5 days or more |

- Server numbers include a margin for slower cores (up to 1.4x slower per core than the Apple M3,
  e.g. AMD EPYC 7742 at 2.25 GHz). Server numbers are estimates, not yet measured.
- The longest single job in short is ~2 h 40 min (in long ~6 h 40 min), so no number of cores makes
  short faster than ~3 h.
- Most of the uncertainty comes from the CNN models of cifar and gtsrb, and in `long` and `full`
  from the S and AB CNN models.

### 3.3 Worst case (every explanation hits its timeout; exact upper bound)

| mode | CPU time | longest job | 6 jobs at once | 48 jobs at once |
|---|---|---|---|---|
| smoke | 5 h 52 min | 6 min | 59 min | 10 min |
| short | 867 h | 2 h 40 min | 145 h | 18 h 26 min |
| long | 3,927 h | 6 h 40 min | 655 h | ~82 h |

## 4. Storage

Space for the results in `data/explanations/` (measured on the test machine):

| mode | size | status |
|---|---|---|
| smoke | 0.5 GB | measured, complete |
| short | 101 GB measured; ~105-110 GB in total | measured without most CNN jobs (the measured gtsrb CNN jobs used 0.24 GB) |
| long | ~270-315 GB | estimated: 2.5x short (20 vs 8 samples), plus ~17-35 GB for the S and AB CNN models |
| full | ~1.4-1.5 TB | estimated: 12.5x short (100 vs 8 samples), plus ~85-175 GB for the S and AB CNN models |

Measured size of short per benchmark (FC models):

| benchmark | size |
|---|---|
| gtsrb | 54.2 GB |
| cifar | 33.7 GB |
| mnist | 11.8 GB |
| loan | 0.34 GB |
| heart_attack | 0.35 GB |
| obesity | 0.35 GB |

- Almost all of the space is the explanation formulas (`*.phi.txt`) of the image models, mainly the
  guide-all and hybrid variants of gtsrb and cifar (single files up to 1.5 GB).
- They are plain text and compress well: `gzip -1` reduces them about 3.5x.
- The CNN results measured so far take about 1-4.5 MB per explanation (gtsrb NS short, mnist
  NS/NAB smoke). The S and AB estimate assumes 3-6 MB per explanation (5,760 in long, 28,800 in full).
- Other outputs are small: `data/explanations_for_interpretation/` 28 MB,
  `data/output/` (figures and tables) ~2 MB, `data/logs/` ~3 MB.

## 5. Notes for the artifact README (from the TACAS'27 AE call)

- The AE call asks for a representative subset that can be reproduced in "about 8 hours on a
  modern notebook". `smoke` (~9 min) fits the smoke test. `short` takes ~2 days and ~105 GB on the
  8-core test machine, so it is above that guideline; a smaller subset (e.g. with `FILTER`, or only
  the tabular benchmarks and mnist) may be needed for the 8-hour target.
- The README must state the tested platforms. Tested here: macOS 26.6.2 on arm64 (Apple M3).
- The reviewer's VM needs enough disk: ~1 GB free for `smoke`, more than ~110 GB for `short`.
