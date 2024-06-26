#!/bin/bash

cd "$(dirname "$0")"

chmod u+x ./install/linux/helper/gen-app-internal-io-structure.sh
chmod u+x ./install/linux/gen-build-install-dir.sh
chmod u+x ./install/linux/gen-host-data-dir.sh
chmod u+x ./install/linux/clean-build-install-dir.sh
chmod u+x ./install/linux/clean-host-data-dir.sh

chmod u+x ./install/linux/install.sh

chmod u+x ./apptainer/0-build-base.sh
chmod u+x ./apptainer/1-build-run.sh
chmod u+x ./apptainer/2-gen-host-data-dir.sh
chmod u+x ./apptainer/3-0-run-test.sh
chmod u+x ./apptainer/3-1-run-test-experiment.sh
chmod u+x ./apptainer/3-2-run-benchmark.sh
chmod u+x ./apptainer/3-3-run-experiment.sh
chmod u+x ./apptainer/clean-sif.sh