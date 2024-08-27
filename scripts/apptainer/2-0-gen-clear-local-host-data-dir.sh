#!/bin/bash

cd "$(dirname "$0")"
cd ../..

if [[ -d host_data/benchmark ]]; then rm -r host_data/benchmark; fi
if [[ -d host_data/experimentation ]]; then rm -r host_data/experimentation; fi

./scripts/install/linux/gen-host-data-dir.sh
