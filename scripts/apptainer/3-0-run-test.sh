#!/bin/bash

cd "$(dirname "$0")"
cd ../../container

apptainer run --nvccli --app test precision-cascade-run.sif