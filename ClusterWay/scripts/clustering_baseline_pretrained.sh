#!/bin/bash

python3 cluster_baseline.py --name deep_way_pretrained --i 0 --seed 42 --clear_file
python3 cluster_baseline.py --name deep_way_pretrained --i 1 --seed 42
python3 cluster_baseline.py --name deep_way_pretrained --i 2 --seed 42

python3 cluster_baseline.py --name deep_way_pretrained --i 0 --seed 42 --curved
python3 cluster_baseline.py --name deep_way_pretrained --i 1 --seed 42 --curved
python3 cluster_baseline.py --name deep_way_pretrained --i 2 --seed 42 --curved