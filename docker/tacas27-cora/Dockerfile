# CORA explanations (cora_explain_dataset, compiled by build_cora_explain) on the MATLAB Runtime,
# for the SpExplAIn experiments with CORA (runme.sh run-cora), e.g. on CSCS Eiger; see README.md.
# Build after data/scripts/tacas27/cora/build.sh (which compiles it into ./cora_explain), from the root of SpExplAIn:
#   docker/build.sh docker/tacas27-cora    (the image spexplain-tacas27-cora:latest)
# The MATLAB Runtime needs no license; its release must be the one of the MATLAB that compiled the application.

ARG MATLAB_RELEASE=r2026a
FROM containers.mathworks.com/matlab-runtime:${MATLAB_RELEASE}

USER root

# What runme.sh and cora1.sh of SpExplAIn use: bash, GNU parallel, Python (to_phi.py), awk, column, timeout
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install --no-install-recommends --yes \
        bash coreutils gawk parallel python3 bsdextrautils procps ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Readable (and the executables executable) by any user (docker run --user), whatever the permissions in the build context
COPY cora_explain /opt/cora_explain
RUN chmod -R a+rX /opt/cora_explain

# The runtime root of the base image (check with: ls /opt/matlabruntime)
ARG MATLAB_RUNTIME_ROOT=/opt/matlabruntime/R2026a
ENV MCR_ROOT=${MATLAB_RUNTIME_ROOT} \
    CORA_BIN=/opt/cora_explain/run_cora_explain_dataset.sh

# The cache of the extracted application, per process; without it, it would go to the read-only image
ENV MCR_CACHE_ROOT=/tmp

# The MATLAB Runtime of the base image runs only with its license accepted
# (see /opt/matlabruntime/R2026a/matlabruntime_license_agreement.pdf)
ENV AGREE_TO_MATLAB_RUNTIME_LICENSE=yes

ENTRYPOINT []
CMD ["/bin/bash"]
