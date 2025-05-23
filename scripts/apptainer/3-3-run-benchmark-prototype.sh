#!/bin/bash

cd "$(dirname "$0")"

./helper/nvidia-arg-help.sh $1 || exit 0
./helper/check-nvidia-arg.sh $1 || exit 1

cd ../../container

HOST_OUTPUT_DIR="../host_data/benchmark/output_data"
CONTR_OUTPUT_DIR="/precision-cascade/install/benchmark/output_data"
BIND_OUTPUT="$HOST_OUTPUT_DIR:$CONTR_OUTPUT_DIR"

if [[ -e $HOST_OUTPUT_DIR ]]; then
    echo "Running benchmark-prototype in container"
    echo "Binding output_data --bind $BIND_OUTPUT"
    apptainer run $1 --writable-tmpfs \
        --bind $BIND_OUTPUT \
        --app benchmark-prototype \
        precision-cascade-run.sif
else
    echo "Missing data dir in $HOST_OUTPUT_DIR"
    exit 1
fi