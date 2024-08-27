#!/bin/bash

cd "$(dirname "$0")"
cd ../../..

mkdir -p host_data

cd host_data

../scripts/install/linux/helper/gen-app-internal-io-structure.sh