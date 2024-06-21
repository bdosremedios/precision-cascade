#!/bin/bash

cd "$(dirname "$0")"
cd ../../..

./scripts/install/general/gen-build-install-dir-structure.sh

cd build
cmake ..
make
cd ..

mv ./build/test/test ./install/test/test
mv ./build/benchmark/benchmark ./install/benchmark/benchmark
mv ./build/experimentation/main/experiment \
   ./install/experimentation/experiment
mv ./build/experimentation/test/test_experiment \
   ./install/experimentation/test/test_experiment
