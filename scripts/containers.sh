#!/usr/bin/env sh

######################################################################
# @author      : tkonuk (tkonuk@draco-oci-login-01)
# @file        : containers
# @created     : Tuesday Oct 24, 2023 16:34:54 PDT
#
# @description : file with holds all the paths required for docker containers
######################################################################

source tokens.sh
CONTAINER="${TKHOME}/pytorch_2309_hf_fa2.sqsh"
CODE="$PROJHOME/code:/code/"
DATASETS="$PROJHOME/datasets:/datasets"

HF_CACHE_DIR="$TKHOME/huggingface/hub:/hub"
EXPERIMENTS="$PROJHOME/experiments:/experiments"

MOUNTS="--container-mounts=$CODE,$DATASETS,$EXPERIMENTS,$HF_CACHE_DIR"
