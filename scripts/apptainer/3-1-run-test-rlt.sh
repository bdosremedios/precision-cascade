#!/bin/bash

cd "$(dirname "$0")"

./helper/nvidia-arg-help.sh $1 || exit 0
./helper/check-nvidia-arg.sh $1 || exit 1

cd ../../container

HOST_TEST_DATA_DIR="../test/data"
CONTR_TEST_DATA_DIR="/precision-cascade/install/test/data"
BIND_TEST_DATA="$HOST_TEST_DATA_DIR:$CONTR_TEST_DATA_DIR"

apptainer run $1 --writable-tmpfs \
    --bind $BIND_TEST_DATA \
    --app test-rlt \
    precision-cascade-run.sif