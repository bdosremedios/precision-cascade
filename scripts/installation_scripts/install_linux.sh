#!/bin/bash

cd "$(dirname "$0")"
cd ../..

mkdir -p build
mkdir -p install

mkdir -p ./install/test
mkdir -p ./install/test/data
mkdir -p ./install/test/data/read_matrices
cp ./test/data/read_matrices/* ./install/test/data/read_matrices
mkdir -p ./install/test/data/solve_matrices
cp ./test/data/solve_matrices/* ./install/test/data/solve_matrices

mkdir -p ./install/benchmark
mkdir -p ./install/benchmark/data
cp ./benchmark/data/* ./install/benchmark/data

mkdir -p ./install/experimentation
mkdir -p ./install/experimentation/main
mkdir -p ./install/experimentation/test
mkdir -p ./install/experimentation/test/data
mkdir -p ./install/experimentation/test/data/test_data
mkdir -p ./install/experimentation/test/data/test_jsons
mkdir -p ./install/experimentation/test/data/test_output

cd build
cmake ..
make

cd ..
mv ./build/test/test ./install/test/test
mv ./build/experimentation/main/experiment \
   ./install/experimentation/main/experiment
mv ./build/experimentation/test/test_experiment \
   ./install/experimentation/test/test_experiment
mv ./build/benchmark/benchmark ./install/benchmark/benchmark
