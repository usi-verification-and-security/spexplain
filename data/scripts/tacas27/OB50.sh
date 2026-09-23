#!/bin/bash
## TACAS'27 experiments for the obesity benchmark (obesity), 50x networks.
## Refer to ./common for the shared driver and the supported env. variables.

NAME=obesity
WIDTH=50
TIMEOUT_PER=${TIMEOUT_PER:-5m}

source "$(dirname "$(realpath "$0")")/common"
