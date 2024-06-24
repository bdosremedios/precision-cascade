#!/bin/bash

cd "$(dirname "$0")"
cd ../../..

if [[ -e data ]]; then rm -r data; fi