# Docker image of SpEXplAIn

# Debian 13: PyTorch 2.x, C++20 compiler, TeX Live with 'standalone' and 'varwidth'
ARG TARGET=debian:trixie

FROM $TARGET
USER root

# make: cmake, build-essential, git (fetches OpenSMT), bison, flex, libgmp-dev
# make marabou: wget (Marabou downloads Boost, OpenBLAS, protobuf, ONNX and pybind11), python3-dev
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
RUN \
    export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) && \
    make && \
    make marabou

# The outputs (e.g. visualizations, also of the reproducibility script) to be inspected on the host,
# intended as the mount point of a host directory (see docker/run.sh)
ENV OUTPUT=$HOME/$PROJECT/data/output
RUN mkdir -p $OUTPUT

ENV PATH=$HOME/$PROJECT:$PATH
