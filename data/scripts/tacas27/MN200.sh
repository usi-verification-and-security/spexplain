#!/bin/bash
## TACAS'27 experiments for the MNIST benchmark (mnist), 200x networks.
## Refer to ./common for the shared driver and the supported env. variables.

NAME=mnist
WIDTH=200
TIMEOUT_PER=${TIMEOUT_PER:-10m}

source "$(dirname "$(realpath "$0")")/common"
