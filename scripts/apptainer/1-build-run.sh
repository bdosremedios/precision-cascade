#!/bin/bash

cd "$(dirname "$0")"
cd ../../container

if [[ -e precision-cascade-run.sif ]]; then rm precision-cascade-run.sif; fi

apptainer build --nvccli precision-cascade-run.sif precision-cascade-run.def