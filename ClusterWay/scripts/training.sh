#!/bin/bash

python3 train.py --i 0 --name deep_way --gpu 0 --clear_file true
python3 train.py --i 1 --name deep_way --gpu 0
python3 train.py --i 2 --name deep_way --gpu 0

python3 train.py --i 0 --name deep_way --curved true --gpu 0
python3 train.py --i 1 --name deep_way --curved true --gpu 0
python3 train.py --i 2 --name deep_way --curved true --gpu 0

python3 train.py --i 0 --name cluster_way --gpu 0
python3 train.py --i 1 --name cluster_way --gpu 0
python3 train.py --i 2 --name cluster_way --gpu 0

python3 train.py --i 0 --name cluster_way --curved true --gpu 0
python3 train.py --i 1 --name cluster_way --curved true --gpu 0
python3 train.py --i 2 --name cluster_way --curved true --gpu 0