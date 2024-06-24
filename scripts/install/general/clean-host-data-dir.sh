#!/bin/bash

cd "$(dirname "$0")"
cd ../../..

if [[ -e host_data ]]; then rm -r host_data; fi