#!/bin/bash
set -e
cd ./feat_extract
git clone https://github.com/princeton-vl/RAFT.git
python3 yolo.py
python3 raft.py
python3 feat_extract.py

/home/wayne/Documents/Progress/RAM/RAFT-master/core