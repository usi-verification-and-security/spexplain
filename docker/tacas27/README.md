# SpExplAIn: TACAS 2027 artifact of the tool paper #9276

This artifact accompanies the tool paper #9276
"SpExplAIn: Trustworthy and Interpretable Space Explanations of Neural Network Classifiers"
submitted to TACAS 2027.
It contains the tool SpExplAIn, which computes and interprets space explanations of the classifications
of neural networks, and everything needed to reproduce the experiments and figures of the paper.

* **Artifact:** TODO(Zenodo DOI link), a self-contained archive with the Docker images below
* **Source code:** https://github.com/usi-verification-and-security/spexplain (tag `tacas27-artifact`)

## Badges

We apply for the **Available**, **Functional** and **Reusable** badges.

* **Available:** the artifact is archived at Zenodo with a DOI (above).
* **Functional:** the artifact computes the explanations of all networks of the paper,
  fully connected and convolutional, and reproduces the performance plots (Figs. 5, 8, 9, 12 and 13)
  and the interpretation figures (Figs. 6, 7, 10 and 11)
  by a single script (see [Reproducing the results of the paper](#reproducing-the-results-of-the-paper)).
* **Reusable:** the tool is open source (MIT license), with a documented command-line interface
  (`spexplain <action> --help`), a hands-on [tutorial](../../tutorial) that runs on new inputs,
  and documented scripts and data formats for adding networks and datasets (see [Reusing the tool](#reusing-the-tool)).

## Contents of the archive

* `README.md`: this file (also included in the image, at `~/spexplain/docker/tacas27/README.md`)
* `LICENSE`: the MIT license
* `spexplain-tacas27_latest.tar`: the Docker image for **amd64** (x86-64) hosts
* `spexplain-tacas27_latest-arm64.tar`: the Docker image for **arm64** hosts (e.g., Apple silicon Macs)

Each image contains the sources of SpExplAIn, already built
(the Explanation Computation Engine, also with the verifier Marabou),
all networks and datasets of the experiments,
and all dependencies of the tool and of the reproducibility script.

## Requirements

* Docker (https://docs.docker.com/get-docker/); on Linux, the commands below may need `sudo`.
  No other software is needed on the host.
* An amd64 or arm64 host with Linux, macOS or Windows.
* Disk: TODO GB for the loaded image, plus TODO GB for the results of the `full` mode.
* Memory: TODO GB.
* No network access is needed after loading the image.
* Optional: a display connected to the container, only to open the pictures in windows
  (see [Showing the pictures in windows](#optional-showing-the-pictures-in-windows));
  otherwise, all pictures are written into files that you open on your computer.

The image also contains [VeriX-plus](https://github.com/minwu-cs/VeriX-plus) (at `~/VeriX-plus`),
for the comparison with it (Figs. 6, 7, 8 and 12),
configured as in the original experiments (its verifier Marabou without Gurobi, so no license is needed).

## Tested platforms

<!-- TODO(authors): the actual machines -->
* amd64: TODO (CPU, cores, memory), TODO (Linux distribution and version), Docker TODO
* arm64: TODO (e.g., MacBook with Apple M-series), macOS TODO, Docker Desktop TODO

## Getting started

All commands below are typed into a terminal of your computer, from the directory of the unpacked archive,
unless they are said to be typed inside the container.

1. Load the image for your host (this takes a few minutes and needs no network):
   ```
   docker load -i spexplain-tacas27_latest.tar          # amd64 (x86-64)
   docker load -i spexplain-tacas27_latest-arm64.tar    # arm64 (e.g., Apple silicon)
   ```
   The image is then available as `spexplain-tacas27:latest` (amd64) or `spexplain-tacas27:latest-arm64` (arm64);
   below, we write `spexplain-tacas27:latest`, replace it if you use arm64.

2. Create a directory for the outputs and start a container from the image:
   ```
   mkdir -p output
   docker run -it --name spexplain-tacas27 -v "$PWD/output:/home/tacas27/spexplain/data/output" spexplain-tacas27:latest
   ```
   (On Windows, use the full path of the directory `output` instead of `$PWD/output`.)
   You are now in a shell inside the container, in the root of the tool (`~/spexplain`).
   The directory `output` of your computer is the directory `data/output` inside the container,
   which is also in the variable `OUTPUT`: all pictures and results that you should look at go there.
   On Linux, if you cannot write into it from the container, allow it by `chmod a+w output` on your computer.

3. Leave the container with `exit`. The container keeps all results;
   continue in it later with:
   ```
   docker start -ai spexplain-tacas27
   ```
   (`docker run` with the same `--name` would fail, because the container already exists;
   remove it with `docker rm spexplain-tacas27` to start from scratch.)

## Smoke test

Inside the container (about TODO minutes in total):

1. Compute the explanations of a toy network and look at one of them:
   ```
   ./spexplain data/models/toy.nnet data/datasets/toy.csv -e $OUTPUT/toy.phi.txt
   ./spexplain interpret $OUTPUT/toy.phi.txt mathexpr data/datasets/toy.csv -i 1 --output-visualization $OUTPUT/toy.pdf
   ```
   The last line printed by the first command is `Done.`;
   the explanation of the first sample is in `output/toy_s1.pdf` on your computer.

2. Run all commands of the [tutorial](../../tutorial) automatically (about a minute):
   ```
   data/scripts/tacas27/tutorial.sh
   ```
   It prints each command with its output and ends with `TUTORIAL PASSED`;
   the pictures and formulas are in `output`.

3. Run the smallest experiments of the paper (about TODO minutes on TODO CPUs):
   ```
   ./reproduce.sh run smoke
   ```
   It ends with the summary of the jobs (all `OK`, or `TIMEOUT` for some explanations)
   and writes the interpretation figures into `output/smoke`.

## Reproducing the results of the paper

All experiments are run by the script `./reproduce.sh` inside the container;
`./reproduce.sh -h` describes all its actions, modes and options.
It runs the experiments in parallel on 80% of the CPUs (the variable `CPU_PERCENTAGE` changes it),
each explanation sequentially, with a timeout per explanation.
With `-n`, it only prints the jobs and their worst-case time, without running anything.

### Modes

The modes differ in the samples of the datasets, the networks and the timeouts:

| Mode    | Samples per dataset | Networks | Time on TODO CPUs |
|---------|---------------------|----------|-------------------|
| `smoke` | the important ones (1-6) | depth up to 6, without CIFAR and GTSRB; CNNs: MNIST only, depth up to 3 | TODO |
| `short` | 8 (the first 8 of `long`) | all (some deep ones left out) | TODO |
| `long`  | 20 | all (some deep ones left out), longer timeouts | TODO |
| `full`  | 100 | all, as in the paper | TODO |

The mode `full` reproduces the results of the paper, with its time limits per explanation
(5 min for the tabular datasets, 10 min for MNIST and CIFAR10, 14 min for GTSRB),
but it takes TODO days on TODO CPUs.
**We recommend the mode `TODO` as a representative subset** (TODO hours on a notebook with TODO cores):
it covers all benchmarks and all variants of the paper with fewer samples and shorter timeouts,
so the absolute numbers differ, but the trends of the paper are visible.
A finished run can be extended later: `--resume` skips the jobs that already finished.

### Commands

For a mode `<mode>` (e.g., `short`), inside the container:

1. Compute the explanations, and the interpretation figures at the end:
   ```
   ./reproduce.sh run <mode>
   ```
   The explanations go to `data/explanations/`, inside the container only.
2. Compute the explanations of VeriX+ for the comparison with it (Figs. 6, 7, 8 and 12),
   on the same networks, samples and timeouts, and remake the interpretation figures with them:
   ```
   ./reproduce.sh run-verix-plus <mode>
   ./reproduce.sh interpret <mode>
   ```
3. Collect the statistics (sizes and run times of the explanations) into tables:
   ```
   STATS_DIR=$OUTPUT/stats ./reproduce.sh collect-stats <mode>
   ```
4. Plot the performance (PAR-2 per network; with the results of step 2, also compared with VeriX+):
   ```
   STATS_DIR=$OUTPUT/stats ./reproduce.sh plot-performance <mode>
   ```

### Where the results of the paper are

All of the following is in the directory `output` of your computer:

| Paper | File |
|-------|------|
| Fig. 5 (PAR-2 of the FC networks with 50 neurons per layer w.r.t. their depth) | `<mode>/fig-5-performance-plots_PAR-2_all_tasks_50.pdf` (and `_200.pdf` for 200 neurons per layer) |
| Fig. 9 (PAR-2 of the narrow CNNs) | `<mode>/fig-9_CNN_performance-plots_PAR-2_narrow-avg-pool.pdf`, `…_narrow-stride2.pdf` |
| Fig. 13 (PAR-2 of the wide CNNs) | TODO (the wide CNNs are not run yet) |
| Figs. 8 and 12 (PAR-2 of SpExplAIn and VeriX+) | `<mode>/fig-8_comparison-plots_PAR-2_ours_vs_verix_plus_all_tasks_50.pdf` and `…_200.pdf`: Fig. 8 is the row of the image datasets of both, Fig. 12 the row of the tabular datasets of `…_50.pdf` |
| the data of these plots | `<mode>/*.csv`; the statistics (sizes, run times, timeouts) in `stats/<mode>/<benchmark>.txt` |
| Fig. 6 (slices of the explanations of heart attack and loan) | `<mode>/fig-6_*.png` |
| Fig. 7 (minimum norm points of MNIST explanations) | `<mode>/fig-7_*.png` (in `smoke`, only the minimal points, without `inv`) |
| Fig. 10 (explanations reduced by `ucore` and `ucore min`) | `<mode>/fig-10_*.png` |
| Fig. 11 (maximum diameters of explanations) | `<mode>/fig-11_*.png` |

**VeriX+.** The parts of VeriX+ in Figs. 6, 7, 8 and 12 come from `./reproduce.sh run-verix-plus <mode>`
(step 2 of the commands); `plot-performance` also writes the run times of VeriX+
as `<mode>/fig-8_comparison-plots_<benchmark>[200]_verix.csv`.
Without these results, the figures are made without VeriX+.

**Disclaimer: the numbers of the samples in Figs. 6, 10 and 11.**
In the paper, these figures (produced by the visualization `fspace`) incorrectly number the samples from 0,
whereas the tool and the artifact number them from 1, as everywhere else (e.g., in Fig. 7 and in the datasets).
Hence, a sample in these figures of the artifact has a number by one greater than in the paper,
e.g., the heart attack sample 97 of Fig. 6 is the sample 98 here (also in the names of the files).
We will correct this in the final version of the paper.

The modes other than `full` use other samples and shorter timeouts,
so their numbers and figures differ from the paper; the samples of the paper's figures
that are not in a mode's dataset are left out of its figures.

## Reusing the tool

* The [tutorial](../../tutorial) introduces the tool by examples, from computing explanations to interpreting them,
  and shows how to try it on other networks and data.
* The main [README](../../README.md) describes all actions and options of the tool;
  each action also prints them with `--help`, e.g., `./spexplain interpret --help`.
* New networks (`.nnet`) and datasets (`.csv`) are used directly by `./spexplain <network> <dataset>`;
  the formats and the organization of the data are described in [data/README.md](../../data/README.md).
* The scripts of the experiments are described in [data/scripts/tacas27/script_all_desc.md](../../data/scripts/tacas27/script_all_desc.md).

## Optional: showing the pictures in windows

The option `--show` of `spexplain interpret` (used by the tutorial) additionally opens the picture in a window.
This requires starting the container connected to the display of your computer:

* **Linux with X11** (or XWayland): allow the access to the display, then start the container with it:
  ```
  xhost +local:
  docker run -it --name spexplain-tacas27 -v "$PWD/output:/home/tacas27/spexplain/data/output" \
      -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro spexplain-tacas27:latest
  ```
  With SELinux (e.g., Fedora), add `--security-opt label=type:container_runtime_t`;
  with errors related to `MIT-SHM`, add `--ipc=host`.
* **macOS:** install and start [XQuartz](https://www.xquartz.org), enable
  *Allow connections from network clients* in its settings (*Security*), restart it, and then:
  ```
  xhost +localhost
  docker run -it --name spexplain-tacas27 -v "$PWD/output:/home/tacas27/spexplain/data/output" \
      -e DISPLAY=host.docker.internal:0 spexplain-tacas27:latest
  ```

If the container already exists, remove it first with `docker rm spexplain-tacas27`.
Without a display, leave out `--show`; the files are written in both cases.

## Building the image from the sources

This is not needed for the evaluation. From the root of the repository:

* Build the image: `docker/build.sh docker/tacas27` (needs network access)
* Run the image: `OUTPUT=<dir> docker/run.sh docker/tacas27`  
  (`<dir>` is mounted at `~/spexplain/data/output`; with `X11=1`, the container is connected to the display,
  as described above; additional arguments of `docker run` can be passed with `DOCKER_RUN_ARGS`)
* Save the image into an archive: `docker/save.sh docker/tacas27`

The build can be adjusted by setting these variables,
e.g., `SPEXPLAIN_GIT_TAG=<branch> docker/build.sh docker/tacas27`:
* `SPEXPLAIN_GIT_TAG`: the branch or tag of the repository (default: `tacas27-artifact`)
* `OPENSMT_GIT_TAG`, `MARABOU_GIT_TAG`: the Git revisions (branches, tags or commits)
  of the dependencies OpenSMT and Marabou (default: pinned to concrete commits)
* `ONNX_ENGINE_GIT_TAG`: the tag of the repository from which the engine for ONNX networks is built
  (default: `onnx-engine-v1`; requires `make onnx` in the repository; an empty value skips it)
* `VERIX_GIT_TAG`, `VERIX_MARABOU_GIT_TAG`: the commits of VeriX-plus and of Marabou within it
  (default: those of the original experiments; an empty `VERIX_GIT_TAG` skips VeriX-plus,
  which makes the image smaller and its build faster)
* `PLATFORM`: the target platform, e.g., `linux/arm64`
* `DOCKER_BUILD_ARGS`: additional arguments of `docker build`, e.g., `--no-cache`

The tool can also be built and run directly on a computer,
see [Building from source](../../README.md#building-from-source).
