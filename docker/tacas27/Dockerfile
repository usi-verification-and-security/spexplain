# Docker image of SpEXplAIn

# Debian 13: PyTorch 2.x, C++20 compiler, TeX Live with 'standalone' and 'varwidth'
ARG TARGET=debian:trixie

FROM $TARGET
USER root

# make: cmake, build-essential, git (fetches OpenSMT), bison, flex, libgmp-dev
# make marabou: wget (Marabou downloads Boost, OpenBLAS, protobuf, ONNX and pybind11), python3-dev
# make onnx: libprotobuf-dev, protobuf-compiler (otherwise protobuf is built from source);
# Marabou does not use them and builds its own tested versions of protobuf, Boost and OpenBLAS
# spexplain: python3
# - interpret: numpy, scipy, matplotlib, pyparsing
# - adversarial neuron activations: pandas, torch
# - list: less (pager)
# data/scripts: bc, parallel, z3, cvc5, column (bsdextrautils),
# gawk (becomes awk instead of mawk, the scripts are developed and tested with GNU awk)
RUN \
    apt-get update -y && \
    apt-get install -y \
        apt-utils \
        ca-certificates \
        make \
        cmake \
        build-essential \
        nano \
        bc \
        gawk \
        git \
        bison \
        flex \
        libgmp-dev \
        libprotobuf-dev \
        protobuf-compiler \
        wget \
        python3 \
        python3-dev \
        python3-numpy \
        python3-scipy \
        python3-matplotlib \
        python3-pyparsing \
        python3-pandas \
        python3-torch \
        parallel \
        bsdextrautils \
        less \
        z3 \
        cvc5 && \
    rm -rf /var/lib/apt/lists/*

# interpret mathexpr rendered to PDF: pdflatex with 'standalone' and 'varwidth' (without docs etc.)
RUN \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        texlive-latex-base \
        texlive-latex-extra && \
    rm -rf /var/lib/apt/lists/*

ENV PROJECT=spexplain
ENV USER=tacas27
ENV HOME=/home/$USER
RUN useradd -m -g users -s /bin/bash $USER

USER $USER
# Matplotlib cache
ENV MPLCONFIGDIR=$HOME/.cache/matplotlib
WORKDIR $HOME

# The branch or tag of the repository (not a commit)
ARG SPEXPLAIN_GIT_TAG=tacas27-artifact
RUN \
    git config --global advice.detachedHead false && \
    git clone https://github.com/usi-verification-and-security/$PROJECT.git --branch $SPEXPLAIN_GIT_TAG --depth 1
WORKDIR $HOME/$PROJECT
# The Git revisions (branches, tags or commits) of the dependencies
ARG OPENSMT_GIT_TAG=195762e8eeb3c5d611fa0aa01a9217e766f2cf5a
ARG MARABOU_GIT_TAG=4dc5e53e5bec22a15b4cfe6b558c02bd15ee14a9
RUN \
    export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) && \
    make OPENSMT_GIT_TAG=$OPENSMT_GIT_TAG && \
    make marabou OPENSMT_GIT_TAG=$OPENSMT_GIT_TAG MARABOU_GIT_TAG=$MARABOU_GIT_TAG

# The engine for ONNX networks is built from a frozen revision (a tag, not included in the shallow clone),
# against the same sources of OpenSMT as above (the frozen revision would otherwise fetch origin/spexplain);
# it requires `make onnx` in the repository, empty ONNX_ENGINE_GIT_TAG skips it
ARG ONNX_ENGINE_GIT_TAG=onnx-engine-v1
RUN \
    if [ -n "$ONNX_ENGINE_GIT_TAG" ]; then \
        git fetch --depth 1 origin tag $ONNX_ENGINE_GIT_TAG && \
        export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) && \
        make onnx ONNX_ENGINE_REV=$ONNX_ENGINE_GIT_TAG \
            ONNX_ONLY_CMAKE_FLAGS=-DFETCHCONTENT_SOURCE_DIR_OPENSMT=$HOME/$PROJECT/build/_deps/opensmt-src; \
    fi

# The outputs (e.g. visualizations, also of the reproducibility script) to be inspected on the host,
# intended as the mount point of a host directory (see docker/run.sh)
ENV OUTPUT=$HOME/$PROJECT/data/output
RUN mkdir -p $OUTPUT

ENV PATH=$HOME/$PROJECT:$PATH
