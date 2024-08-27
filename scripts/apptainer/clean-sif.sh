#!/bin/bash

cd "$(dirname "$0")"
cd ../../container

if [[ -e precision-cascade-base.sif ]]; then rm precision-cascade-base.sif; fi
if [[ -e precision-cascade-run.sif ]]; then rm precision-cascade-run.sif; fi