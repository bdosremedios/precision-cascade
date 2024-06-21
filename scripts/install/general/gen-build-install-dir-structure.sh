#!/bin/bash

cd "$(dirname "$0")"
cd ../../..

mkdir -p build
mkdir -p install

mkdir -p ./install/test
mkdir -p ./install/test/data
mkdir -p ./install/test/data/read_matrices
cp ./test/data/read_matrices/* ./install/test/data/read_matrices
mkdir -p ./install/test/data/solve_matrices
cp ./test/data/solve_matrices/* ./install/test/data/solve_matrices

mkdir -p ./install/benchmark

mkdir -p ./install/experimentation
mkdir -p ./install/experimentation/test
mkdir -p ./install/experimentation/test/data
mkdir -p ./install/experimentation/test/data/test_data
cp ./experimentation/test/data/test_data/* \
   ./install/experimentation/test/data/test_data
mkdir -p ./install/experimentation/test/data/test_jsons
cp ./experimentation/test/data/test_jsons/* \
   ./install/experimentation/test/data/test_jsons
mkdir -p ./install/experimentation/test/data/test_output