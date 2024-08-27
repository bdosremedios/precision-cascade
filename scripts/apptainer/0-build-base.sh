#!/bin/bash

cd "$(dirname "$0")"
cd ../../container

if [[ -e precision-cascade-base.sif ]]; then rm precision-cascade-base.sif; fi

apptainer build precision-cascade-base.sif precision-cascade-base.def