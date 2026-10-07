#!/bin/bash
## TACAS'27 experiments for the GTSRB benchmark (gtsrb), 50x networks.
## Refer to ./common for the shared driver and the supported env. variables.

NAME=gtsrb
WIDTH=50
TIMEOUT_PER=${TIMEOUT_PER:-15m}

source "$(dirname "$(realpath "$0")")/common"
