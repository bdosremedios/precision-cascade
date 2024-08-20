#!/bin/bash

cd "$(dirname "$0")"

chmod u+x ./install/linux/helper/gen-app-internal-io-structure.sh
chmod u+x ./install/linux/gen-build-install-dir.sh
chmod u+x ./install/linux/gen-host-data-dir.sh
chmod u+x ./install/linux/clean-build-install-dir.sh
chmod u+x ./install/linux/clean-host-data-dir.sh

chmod u+x ./install/linux/install.sh

chmod u+x ./apptainer/helper/nvidia-arg-help.sh
chmod u+x ./apptainer/helper/check-nvidia-arg.sh
chmod u+x ./apptainer/0-build-base.sh
chmod u+x ./apptainer/1-build-run.sh
chmod u+x ./apptainer/2-0-gen-clear-local-host-data-dir.sh
chmod u+x ./apptainer/2-1-gen-clear-remote-ln-host-data-dir.sh
chmod u+x ./apptainer/3-0-run-test.sh
chmod u+x ./apptainer/3-1-run-test-rlt.sh
chmod u+x ./apptainer/3-2-run-test-experiment.sh
chmod u+x ./apptainer/4-0-run-experiment.sh
chmod u+x ./apptainer/4-1-run-benchmark.sh
chmod u+x ./apptainer/clean-sif.sh