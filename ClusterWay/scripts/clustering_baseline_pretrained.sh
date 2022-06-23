#!/bin/bash

python3 cluster_baseline.py --name deep_way_pretrained --i 0 --seed 42 --clear_file true
python3 cluster_baseline.py --name deep_way_pretrained --i 1 --seed 42
python3 cluster_baseline.py --name deep_way_pretrained --i 2 --seed 42

python3 cluster_baseline.py --name deep_way_pretrained --i 0 --seed 42 --curved true
python3 cluster_baseline.py --name deep_way_pretrained --i 1 --seed 42 --curved true
python3 cluster_baseline.py --name deep_way_pretrained --i 2 --seed 42 --curved true