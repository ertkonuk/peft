#!/usr/bin/env sh

######################################################################
# @author      : tkonuk (tkonuk@draco-oci-login-01)
# @file        : containers
# @created     : Tuesday Oct 24, 2023 16:34:54 PDT
#
# @description : file with holds all the paths required for docker containers
######################################################################
WANDB=""
HF_TOKEN=""

CONTAINER="nvcr.io/nvidia/pytorch:23.09-py3"
USER=tkonuk
PROJHOME="/lustre/fsw/portfolios/llmservice/users/${USER}/tied-lora"
CODE="$PROJHOME/code:/code/"
DATASETS="$PROJHOME/datasets:/datasets"
MODELS="$PROJHOME/trained_models:/trained_models,$PROJHOME/pretrained_models:/pretrained_models"

HF_CACHE_DIR="$PROJHOME/huggingface/hub:/hub"
REQS="$PROJHOME/requirements.txt:/requirements/requirements.txt"
EXPERIMENTS="$PROJHOME/experiments:/experiments"
SCRIPTS="$PROJHOME/launch_scripts:/launch_scripts,$PROJHOME/eval_scripts:/eval_scripts"

MOUNTS="--container-mounts=$CODE,$DATASETS,$MODELS,$EXPERIMENTS,$SCRIPTS,$REQS,$HF_CACHE_DIR"
