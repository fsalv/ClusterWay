#!/bin/bash

python3 cluster_baseline.py --i 0 --clear_file
python3 cluster_baseline.py --i 1
python3 cluster_baseline.py --i 2

python3 cluster_baseline.py --i 0 --curved
python3 cluster_baseline.py --i 1 --curved
python3 cluster_baseline.py --i 2 --curved