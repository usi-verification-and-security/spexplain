# CORA explanations for the SpExplAIn experiments, on CSCS Eiger

`runme.sh run-cora` (`data/scripts/tacas27`, branch `tacas27-cora`) runs CORA (`neuralNetwork/explain`,
https://github.com/TUMcps/CORA) by `cora_explain_dataset` (`data/scripts/tacas27/cora`), compiled into
a standalone application on the free MATLAB Runtime, so the cluster needs no MATLAB license, network access
or login. CORA is its release `v2026.1.0` with `cora-explain.patch`, which adds the options `InputBounds`
(the perturbation within [0,1]) and `Plot` (no figures) of `explain`; there is no fork of CORA.

The image and the steps 3 and 4 aim at the cluster Eiger of CSCS (the Swiss National Supercomputing Centre,
https://www.cscs.ch): its Container Engine (the environments in `~/.edf`), podman and enroot, Slurm
and `$SCRATCH`. The steps 1 and 2 (the compilation, the image and its local test) work on any machine.

## 1. Compile (on a machine with MATLAB)

Requirements: MATLAB R2026a, the same update as the MATLAB Runtime of the image
(`containers.mathworks.com/matlab-runtime:r2026a`, Update 5 so far), with MATLAB Compiler and the products
of CORA for neural networks: Deep Learning Toolbox, Optimization Toolbox, Statistics and Machine Learning
Toolbox and Symbolic Math Toolbox (e.g. `mpm install --release=R2026a --products MATLAB_Compiler ...`).

```bash
data/scripts/tacas27/cora/build.sh <CORA dir>
```

If `<CORA dir>` does not exist, it clones CORA `v2026.1.0` into it and applies the patch
(an existing one must be at that release with the patch applied).
The application goes to `docker/tacas27-cora/cora_explain/` (`cora_explain_dataset` and
`run_cora_explain_dataset.sh`), which `runme.sh run-cora` then uses by default; locally,
the MATLAB installation works as the runtime.

## 2. Build the image

```bash
docker/build.sh docker/tacas27-cora       # the image spexplain-tacas27-cora:latest
docker run --rm spexplain-tacas27-cora ls /opt/matlabruntime    # the runtime root, as MATLAB_RUNTIME_ROOT
docker/save.sh docker/tacas27-cora        # spexplain-tacas27-cora_latest.tar
```

Keep the archive of the image used for the results: the application runs only with the MATLAB Runtime
of the release and update that compiled it.

To test it locally, in the SpExplAIn checkout, as your user (otherwise the results are owned by root):

```bash
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $PWD:$PWD -w $PWD/data/scripts/tacas27 \
    spexplain-tacas27-cora env FILTER='^(heart_attack|mnist)/50x1/' ./runme.sh run-cora smoke
```

## 3. Import it on CSCS Eiger

```bash
# locally: the archive goes to the scratch of Eiger ($SCRATCH there)
scp spexplain-tacas27-cora_latest.tar eiger:/ritom/scratch/cscs/<user>/
# on Eiger, on a compute node (podman and enroot are not for the login nodes)
srun -A u17 -p debug --pty bash
cd $SCRATCH
podman load -i spexplain-tacas27-cora_latest.tar
# the image itself goes to the persistent home (the archive can stay on the scratch)
mkdir -p ~/docker
enroot import -x mount -o ~/docker/spexplain-tacas27-cora_latest.sqsh podman://spexplain-tacas27-cora:latest
```

Then copy `cora-matlab.toml` to `~/.edf/` (its `image` and mounts use `${HOME}` and `${SCRATCH}`:
the Container Engine expands variables only with the braces).
The SpExplAIn checkout must be under `$HOME` or `$SCRATCH` (its `mounts`).
Refer to https://docs.cscs.ch/software/container-engine/ for the details.

## 4. Run on CSCS Eiger

In the SpExplAIn checkout (branch `tacas27-cora`):

```bash
cd data/scripts/tacas27
# a short test in the debug partition first
DEBUG=1 ENV=cora-matlab CPU_PERCENTAGE=50 WRAPPER=~/slurm/sbatch.sh FILTER='^mnist/50x1/' ./runme.sh run-cora smoke
# all; resubmit with --resume after the 24h limit
ENV=cora-matlab CPU_PERCENTAGE=50 WRAPPER=~/slurm/sbatch.sh ./runme.sh run-cora full
```

`~/slurm/sbatch.sh` is only an example: `WRAPPER` is whatever script in your home on the cluster submits
the command by Slurm (`sbatch`), e.g. in the environment `ENV` of the Container Engine (`--environment`).
Batches by disjoint `FILTER`s can run at the same time: each gets its own log directory (see `LOG_DIR`
in `runme.sh`); resubmit each with the same `FILTER` and `--resume`.

The image sets `CORA_BIN` and `MCR_ROOT`, so `cora/cora1.sh` uses the compiled application.
It also sets `AGREE_TO_MATLAB_RUNTIME_LICENSE=yes`, without which the MATLAB Runtime of the base image does not
start: building and running the image accepts the MATLAB Runtime license
(`/opt/matlabruntime/R2026a/matlabruntime_license_agreement.pdf` in the image).
Each process of the MATLAB Runtime (without Java) takes about 0.5–1 GB of memory, so a node (256 GB) fits
one job per core: `CPU_PERCENTAGE=50` of its 256 hardware threads, as in the commands above
(each CORA job runs on its own core, see `cora/cora1.sh`).
