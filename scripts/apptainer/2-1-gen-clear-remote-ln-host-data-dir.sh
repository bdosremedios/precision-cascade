#!/bin/bash

cd "$(dirname "$0")"

if [ "$1" == "--help" ] || [ "$1" == "-H" ]; then
    echo "Provide absolute path of remote host directory to store data in"
    exit 0
elif [[ ! -d $1 ]]; then
    echo "Remote host directory does not exist"
    exit 1
fi

REMOTE_DATA_DIR="$1"
SCRIPT_ABS_DIR="$(pwd)"

# Create internal structure in REMOTE_DATA_DIR if does not exist
mkdir -p $REMOTE_DATA_DIR
cd $REMOTE_DATA_DIR
"$SCRIPT_ABS_DIR/../install/linux/helper/gen-app-internal-io-structure.sh"

# Link internal structure in REMOTE_DATA_DIR to host_data in precision-cascade
# location
cd $SCRIPT_ABS_DIR
cd ../..

mkdir -p host_data
if [[ -d host_data/benchmark ]]; then rm -r host_data/benchmark; fi
if [[ -d host_data/experimentation ]]; then rm -r host_data/experimentation; fi
cd host_data

if [[ -L benchmark ]]; then rm benchmark; fi
ln -s "$REMOTE_DATA_DIR/benchmark/" benchmark

if [[ -L experimentation ]]; then rm experimentation; fi
ln -s "$REMOTE_DATA_DIR/experimentation/" experimentation
