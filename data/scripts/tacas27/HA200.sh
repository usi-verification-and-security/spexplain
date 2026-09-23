#!/bin/bash
## TACAS'27 experiments for the heart attack benchmark (heart_attack), 200x networks.
## Refer to ./common for the shared driver and the supported env. variables.

NAME=heart_attack
WIDTH=200
TIMEOUT_PER=${TIMEOUT_PER:-5m}

source "$(dirname "$(realpath "$0")")/common"
