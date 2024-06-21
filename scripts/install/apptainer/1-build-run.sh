#!/bin/bash

cd "$(dirname "$0")"
cd ../../container

apptainer build --nvccli precision-cascade-run.sif precision-cascade-run.def