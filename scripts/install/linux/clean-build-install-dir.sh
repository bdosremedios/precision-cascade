#!/bin/bash

cd "$(dirname "$0")"
cd ../../..

if [[ -e build ]]; then rm -r build; fi
if [[ -e install ]]; then rm -r install; fi