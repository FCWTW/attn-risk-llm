#!/bin/bash
set -e
cd "$(dirname "$0")"
cd ./feat_extract
if [ ! -d "RAFT" ]; then
    echo "Copying the RAFT repository..."
    git clone https://github.com/princeton-vl/RAFT.git
else
    echo "RAFT repository detected"
fi

python3 preprocess.py
python3 feat_extract.py