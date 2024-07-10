#!/bin/bash

if [ "$1" == "--help" ] || [ "$1" == "-H" ]; then
    echo "Provide argument --nv or --nvccli"
    exit 0
elif [ "$1" != "--nv" ] && [ "$1" != "--nvccli" ]; then
    echo "Invalid argument given must be --nv or --nvccli"
    exit 1
else
    exit 0
fi