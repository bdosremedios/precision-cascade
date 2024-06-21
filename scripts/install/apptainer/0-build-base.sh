#!/bin/bash

cd "$(dirname "$0")"
cd ../../container

apptainer build precision-cascade-base.sif precision-cascade-base.def