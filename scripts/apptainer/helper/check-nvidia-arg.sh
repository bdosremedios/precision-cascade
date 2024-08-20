#!/bin/bash

if [ "$1" != "--nv" ] && [ "$1" != "--nvccli" ]; then
    echo "Invalid argument given must be --nv or --nvccli"
    exit 1
else
    exit 0
fi