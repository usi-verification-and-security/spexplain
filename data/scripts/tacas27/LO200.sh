#!/bin/bash
## TACAS'27 experiments for the loan benchmark (loan), 200x networks.
## Refer to ./common for the shared driver and the supported env. variables.

NAME=loan
WIDTH=200
TIMEOUT_PER=${TIMEOUT_PER:-5m}

source "$(dirname "$(realpath "$0")")/common"
