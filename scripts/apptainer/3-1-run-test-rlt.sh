#!/bin/bash

cd "$(dirname "$0")"

./helper/nvidia-arg-help.sh $1 || exit 0
./helper/check-nvidia-arg.sh $1 || exit 1

cd ../../container

apptainer run $1 --writable-tmpfs --app test-rlt precision-cascade-run.sif