# SpEXplAIn - Docker image

The image contains the tool SpEXplAIn built from its sources
(the Explanation Computation Engine, also with the verifier Marabou)
together with all dependencies of the main script `spexplain`,
including the Explanation Interpreter (`spexplain interpret`).

## Building and running

From the root of the repository:

* Build the image: `docker/build.sh docker/tacas27`  
  (the build can be adjusted, see [Build options](#build-options))
* Run the image: `OUTPUT=<dir> docker/run.sh docker/tacas27`  
  (the host directory `<dir>` is mounted at `~/spexplain/data/output` inside the container,
  which is also in the variable `OUTPUT`)

Inside the container, the tool is located at `~/spexplain`
and the script `spexplain` is available in `PATH`.
To get started, follow the [tutorial](../../tutorial), which is designed for this image.
The experiments of the paper are reproduced by `./reproduce.sh` (see `./reproduce.sh -h`),
which writes its figures into `data/output`, i.e., into the mounted directory `<dir>`.

## Visualizations

All visualizations are written into files,
which can be inspected on the host using the mounted output directory, for example:

```
spexplain data/models/toy.nnet data/datasets/toy.csv -e $OUTPUT/phi.txt
spexplain interpret pixel data/datasets/mnist/mnist_s100_scaled.csv -i 1 --output-visualization $OUTPUT/sample.png
spexplain interpret $OUTPUT/phi.txt mathexpr data/datasets/toy.csv --output-visualization $OUTPUT/phi.pdf
```

## Showing visualizations

The option `--show` additionally shows the visualization in a window,
which requires connecting the container to a display of the host
(this is optional, the files in `$OUTPUT` are produced anyway).

On Linux with X11, or with XWayland under Wayland, run the image with:
`X11=1 OUTPUT=<dir> docker/run.sh docker/tacas27`  
(it passes `DISPLAY`, mounts the X11 socket directory `/tmp/.X11-unix`,
and an authorization cookie of the display created with `xauth`).
Additional arguments of `docker run` can be passed with the variable `DOCKER_RUN_ARGS`.
Possible issues:
* The cookie is readable only by the host user and it is read by the user inside the container with UID 1000;
  if the UID of the host user differs, or if `xauth` is not available,
  allow the access to the display instead, e.g., with `xhost +local:`.
* With SELinux (e.g., Fedora), the container is not allowed to connect to the display by default:
  use `DOCKER_RUN_ARGS='--security-opt label=type:container_runtime_t'`.
* Errors related to `MIT-SHM`: use `DOCKER_RUN_ARGS=--ipc=host`.
* There must be an X server (e.g., XWayland must be enabled under Wayland).
* With Docker Desktop on Linux (which runs the containers in a virtual machine),
  the socket directory of the host is not visible to the containers.

On macOS with Docker Desktop, the display is connected via TCP using [XQuartz](https://www.xquartz.org):
install and start XQuartz, enable *Allow connections from network clients* in its settings (*Security*),
restart it, allow the access with `xhost +localhost`, and then run the image with:
`X11=1 OUTPUT=<dir> docker/run.sh docker/tacas27`.

## Build options

The build of the image can be adjusted by setting these variables,
e.g., `SPEXPLAIN_GIT_TAG=<branch> docker/build.sh docker/tacas27`:
* `SPEXPLAIN_GIT_TAG`: the branch or tag of the repository (default: `tacas27-artifact`)
* `OPENSMT_GIT_TAG`, `MARABOU_GIT_TAG`: the Git revisions (branches, tags or commits)
  of the dependencies OpenSMT and Marabou (default: pinned to concrete commits)
* `ONNX_ENGINE_GIT_TAG`: the tag of the repository from which the engine for ONNX networks is built
  (default: `onnx-engine-v1`; requires `make onnx` in the repository; an empty value skips it)
* `PLATFORM`: the target platform, e.g., `linux/arm64`
* `DOCKER_BUILD_ARGS`: additional arguments of `docker build`, e.g., `--no-cache`
