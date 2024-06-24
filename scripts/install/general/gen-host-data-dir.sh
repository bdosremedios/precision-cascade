#!/bin/bash

cd "$(dirname "$0")"
cd ../../..

mkdir -p host_data

mkdir -p host_data/benchmark
mkdir -p host_data/benchmark/data

mkdir -p host_data/experiment
mkdir -p host_data/experiment/data
mkdir -p host_data/experiment/input
mkdir -p host_data/experiment/output