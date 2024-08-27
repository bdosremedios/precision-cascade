#!/bin/bash

if [ "$1" == "--help" ] || [ "$1" == "-H" ]; then
    echo "Provide argument --nv or --nvccli"
    exit 1
else
    exit 0
fi