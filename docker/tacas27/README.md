# SpEXplAIn - Docker image

The image contains the tool SpEXplAIn built from its sources
(the Explanation Computation Engine, also with the verifier Marabou)
together with all dependencies of the main script `spexplain`,
including the Explanation Interpreter (`spexplain interpret`).

## Building and running

From the root of the repository:

* Build the image: `docker/build.sh docker/tacas27`  
  (the variable `SPEXPLAIN_GIT_TAG` selects the branch or tag of the repository, default: `tacas27-artifact`;
  the variable `PLATFORM` selects the target platform, e.g., `linux/arm64`;
  the Git revisions of the dependencies OpenSMT and Marabou are pinned to concrete commits,
  which can be changed with `--build-arg OPENSMT_GIT_TAG=...` and `--build-arg MARABOU_GIT_TAG=...`
  using `docker build` directly)
* Run the image: `OUTPUT=<dir> docker/run.sh docker/tacas27`  
  (the host directory `<dir>` is mounted at `~/spexplain/data/output` inside the container,
  which is also in the variable `OUTPUT`)

Inside the container, the tool is located at `~/spexplain`
and the script `spexplain` is available in `PATH`.
To get started, follow the [tutorial](../../tutorial), which is designed for this image.

## Visualizations

All visualizations are written into files,
which can be inspected on the host using the mounted output directory, for example:

```
spexplain data/models/toy.nnet data/datasets/toy.csv -e $OUTPUT/phi.txt
spexplain interpret pixel data/datasets/mnist/mnist_s100_scaled.csv -i 1 --output-visualization $OUTPUT/sample.png
spexplain interpret $OUTPUT/phi.txt mathexpr data/datasets/toy.csv --output-visualization $OUTPUT/phi.pdf
```

The option `--show` is not supported inside the container, since it requires a display.
