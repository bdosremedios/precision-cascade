#!/bin/bash

cd "$(dirname "$0")"

chmod u+x ./general/gen-build-install-dir-structure.sh
chmod u+x ./general/gen-data-dir-structure.sh
chmod u+x ./general/clean-build-install.sh

chmod u+x ./apptainer/0-build-base.sh
chmod u+x ./apptainer/1-build-run.sh
chmod u+x ./apptainer/2-gen-data-dirs.sh

chmod u+x ./linux/install.sh