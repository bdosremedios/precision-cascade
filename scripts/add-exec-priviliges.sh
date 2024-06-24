#!/bin/bash

cd "$(dirname "$0")"

chmod u+x ./install/general/gen-build-install-dir.sh
chmod u+x ./install/general/gen-data-dir.sh
chmod u+x ./install/general/clean-build-install-dir.sh
chmod u+x ./install/general/clean-data-dir.sh
chmod u+x ./install/general/clean-dir.sh

chmod u+x ./install/linux/install.sh

chmod u+x ./apptainer/0-build-base.sh
chmod u+x ./apptainer/1-build-run.sh
chmod u+x ./apptainer/2-gen-data-dirs.sh
chmod u+x ./apptainer/3-0-run-test.sh
chmod u+x ./apptainer/3-1-run-test-experiment.sh
chmod u+x ./apptainer/clean-sif.sh