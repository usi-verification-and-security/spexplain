# Docker image of SpExplAIn, with VeriX-plus for its experiments

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

# interpret mathexpr rendered to PDF: pdflatex with 'standalone' and 'varwidth' (without docs etc.);
# plot-performance: Matplotlib with LaTeX (usetex) requires cm-super (type1ec.sty) and dvipng
# (its layout renders the text with Agg, also for PDF)
RUN \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        texlive-latex-base \
        texlive-latex-extra \
        cm-super-minimal \
        dvipng && \
    rm -rf /var/lib/apt/lists/*

# Only for --show with a display connected (see docker/run.sh, X11=1):
# python3-tk (interactive backend of matplotlib) for pixel and fspace,
# xdg-utils and mupdf (PDF viewer) for mathexpr
RUN \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3-tk \
        xdg-utils \
        mupdf && \
    rm -rf /var/lib/apt/lists/*

ENV PROJECT=spexplain
ENV USER=tacas27
ENV HOME=/home/$USER
RUN useradd -m -g users -s /bin/bash $USER

USER $USER
# Matplotlib cache
ENV MPLCONFIGDIR=$HOME/.cache/matplotlib
WORKDIR $HOME
# xdg-open opens PDFs in mupdf, there is no desktop environment
RUN xdg-mime default mupdf.desktop application/pdf

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

# VeriX-plus, for its experiments (data/scripts/tacas27/runme.sh run-verix-plus), configured as the original ones:
# with the patch for binary classification (of the original experiments), and with Marabou without Gurobi;
# empty VERIX_GIT_TAG skips it
ENV VERIX_DIR=$HOME/VeriX-plus
# The Git revisions (commits) of VeriX-plus and of Marabou within it, as used in the original experiments
ARG VERIX_GIT_TAG=5efb63607da04155214c8a2e3374b954759de642
ARG VERIX_MARABOU_GIT_TAG=1c2f4788c32e2f4e407c356b763a8025c5578722
# uv installs the Python of VeriX-plus (3.8) and its pinned packages
COPY --from=ghcr.io/astral-sh/uv:0.5.31 /uv /usr/local/bin/uv
# VERIX_DIR is made by the user, who then clones and builds in it (not by COPY, which would apply --chmod to it)
RUN mkdir -p $VERIX_DIR
# Readable by the user, whatever the permissions in the build context
COPY --chmod=644 verix-requirements.txt verix-tabular.patch $VERIX_DIR/
# maraboupy (MarabouCore) for the Python of VeriX-plus; the MILP queries are then 'sat' unless refuted
# by the preprocessing of Marabou, as in the original experiments
RUN \
    if [ -n "$VERIX_GIT_TAG" ]; then \
        git init -q $VERIX_DIR && \
        git -C $VERIX_DIR fetch -q --depth 1 https://github.com/minwu-cs/VeriX-plus.git $VERIX_GIT_TAG && \
        git -C $VERIX_DIR checkout -q FETCH_HEAD && \
        git -C $VERIX_DIR apply $VERIX_DIR/verix-tabular.patch && \
        git init -q $VERIX_DIR/Marabou && \
        git -C $VERIX_DIR/Marabou fetch -q --depth 1 https://github.com/NeuralNetworkVerification/Marabou.git $VERIX_MARABOU_GIT_TAG && \
        git -C $VERIX_DIR/Marabou checkout -q FETCH_HEAD && \
        uv venv -q --python 3.8 $VERIX_DIR/venv && \
        VIRTUAL_ENV=$VERIX_DIR/venv uv pip install -q --no-cache \
            --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
            -r $VERIX_DIR/verix-requirements.txt && \
        cmake -S $VERIX_DIR/Marabou -B $VERIX_DIR/Marabou/build -DCMAKE_BUILD_TYPE=Release \
            -DENABLE_GUROBI=OFF -DBUILD_PYTHON=ON -DPYTHON_EXECUTABLE=$VERIX_DIR/venv/bin/python \
            -DRUN_UNIT_TEST=OFF -DRUN_REGRESS_TEST=OFF -DRUN_SYSTEM_TEST=OFF -DRUN_PYTHON_TEST=OFF && \
        cmake --build $VERIX_DIR/Marabou/build --parallel $(nproc) && \
        ! ldd $VERIX_DIR/Marabou/maraboupy/MarabouCore*.so | grep -q libgurobi && \
        $VERIX_DIR/venv/bin/python -c "import sys; sys.path.insert(0, '$VERIX_DIR/Marabou'); from maraboupy import Marabou"; \
    fi

# The outputs (e.g. visualizations, also of reproduce.sh) to be inspected on the host,
# intended as the mount point of a host directory (see docker/run.sh)
ENV OUTPUT=$HOME/$PROJECT/data/output
RUN mkdir -p $OUTPUT

# The reproducibility script of TACAS'27 in the root of the repository (ignored by git)
RUN ln -s data/scripts/tacas27/runme.sh reproduce.sh && echo /reproduce.sh >>.git/info/exclude

ENV PATH=$HOME/$PROJECT:$PATH
