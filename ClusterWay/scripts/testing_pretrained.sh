#!/bin/bash

python3 test.py --i 0 --name deep_way_pretrained --seed 42 --clear_file true
python3 test.py --i 1 --name deep_way_pretrained --seed 42
python3 test.py --i 2 --name deep_way_pretrained --seed 42

python3 test.py --i 0 --name deep_way_pretrained --curved true --seed 42
python3 test.py --i 1 --name deep_way_pretrained --curved true --seed 42
python3 test.py --i 2 --name deep_way_pretrained --curved true --seed 42

python3 test.py --i 0 --name cluster_way_pretrained --seed 42
python3 test.py --i 1 --name cluster_way_pretrained --seed 42
python3 test.py --i 2 --name cluster_way_pretrained --seed 42

python3 test.py --i 0 --name cluster_way_pretrained --curved true --seed 42
python3 test.py --i 1 --name cluster_way_pretrained --curved true --seed 42
python3 test.py --i 2 --name cluster_way_pretrained --curved true --seed 42