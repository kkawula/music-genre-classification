#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"

cd "$DATA_DIR"

curl -O https://os.unil.cloud.switch.ch/fma/fma_medium.zip

echo "c67b69ea232021025fca9231fc1c7c1a063ab50b  fma_medium.zip" | sha1sum -c -

7z x -aos fma_medium.zip
