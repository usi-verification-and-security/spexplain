# Created by Tomas Kolarik
# Eternal glory to him

ARG TARGET=debian:latest

FROM $TARGET
USER root

RUN \
    apt-get update -y && \
    apt-get install -y \
        apt-utils \
        make \
        cmake \
        build-essential \
        nano \
        bc \
        git \
        bison \
        flex \
        libgmp-dev \
        wget \
        python3-dev \
        parallel \
        z3 \
        cvc5

ENV PROJECT=xspace
ENV USER=cav25
ENV HOME=/home/$USER
RUN useradd -m -g users -s /bin/bash $USER
USER $USER
WORKDIR $HOME

ENV TAG=cav25-updated
RUN \
    git config --global advice.detachedHead false && \
    git clone --recurse-submodules https://github.com/usi-verification-and-security/$PROJECT.git --branch $TAG --depth 1
WORKDIR $PROJECT
RUN \
    export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) && \
    make && \
    make marabou
# RUN make install

COPY --chown=$USER:users README.md LICENSE runme.sh $HOME/$PROJECT
