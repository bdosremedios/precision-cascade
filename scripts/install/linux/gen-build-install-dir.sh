#!/bin/bash

cd "$(dirname "$0")"
cd ../../..

mkdir -p build
mkdir -p install

cd install

mkdir -p ./test
mkdir -p ./test/data
mkdir -p ./test/data/read_matrices
cp ../test/data/read_matrices/* ./test/data/read_matrices
mkdir -p ./test/data/solve_matrices
cp ../test/data/solve_matrices/* ./test/data/solve_matrices

../scripts/install/linux/helper/gen-app-internal-io-structure.sh

mkdir -p ./experimentation/test
mkdir -p ./experimentation/test/data
mkdir -p ./experimentation/test/data/test_data
cp ../experimentation/test/data/test_data/* \
   ./experimentation/test/data/test_data
mkdir -p ./experimentation/test/data/test_jsons
cp ../experimentation/test/data/test_jsons/* \
   ./experimentation/test/data/test_jsons
mkdir -p ./experimentation/test/data/test_output