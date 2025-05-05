#!/bin/bash

cd "$(dirname "$0")"

./helper/nvidia-arg-help.sh $1 || exit 0
./helper/check-nvidia-arg.sh $1 || exit 1

cd ../../container

HOST_EXPERIMENTATION_TEST_DATA_DIR="../experimentation/test/data"
CONTR_EXPERIMENTATION_TEST_DATA_DIR="/precision-cascade/install/experimentation/test/data"
BIND_EXPERIMENTATION_TEST_DATA="$HOST_EXPERIMENTATION_TEST_DATA_DIR:$CONTR_EXPERIMENTATION_TEST_DATA_DIR"

apptainer run $1 --writable-tmpfs \
    --bind $BIND_EXPERIMENTATION_TEST_DATA \
    --app test-experiment \
    precision-cascade-run.sif