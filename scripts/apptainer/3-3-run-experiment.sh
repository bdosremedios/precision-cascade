#!/bin/bash

cd "$(dirname "$0")"
cd ../../container

HOST_DATA_DIR="../host_data/experimentation/matrix_data"
HOST_INPUT_DIR="../host_data/experimentation/input_specs"
HOST_OUTPUT_DIR="../host_data/experimentation/output_data"

CONTR_DATA_DIR="/precision-cascade/install/experimentation/matrix_data"
CONTR_INPUT_DIR="/precision-cascade/install/experimentation/input_specs"
CONTR_OUTPUT_DIR="/precision-cascade/install/experimentation/output_data"

BIND_DATA="$HOST_DATA_DIR:$CONTR_DATA_DIR"
BIND_INPUT="$HOST_INPUT_DIR:$CONTR_INPUT_DIR"
BIND_OUTPUT="$HOST_OUTPUT_DIR:$CONTR_OUTPUT_DIR"

if [[ -e $HOST_DATA_DIR && -e $HOST_INPUT_DIR && -e $HOST_OUTPUT_DIR ]]; then
    echo "Running experiment in container"
    echo "Binding matrix_data --bind $BIND_DATA"
    echo "Binding input_specs --bind $BIND_INPUT"
    echo "Binding output_data --bind $BIND_OUTPUT"
    apptainer run --nvccli \
        --bind "$BIND_DATA,$BIND_INPUT,$BIND_OUTPUT" \
        --app experiment \
        precision-cascade-run.sif
else
    echo "Missing data dir in $HOST_DATA_DIR or $HOST_INPUT_DIR or \
$HOST_OUTPUT_DIR"
    exit 1
fi