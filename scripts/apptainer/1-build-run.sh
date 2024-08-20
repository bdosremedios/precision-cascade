#!/bin/bash

cd "$(dirname "$0")"

./helper/nvidia-arg-help.sh $1 || exit 0
./helper/check-nvidia-arg.sh $1 || exit 1

cd ../../container

if [[ -e precision-cascade-run.sif ]]; then rm precision-cascade-run.sif; fi

apptainer build $1 precision-cascade-run.sif precision-cascade-run.def