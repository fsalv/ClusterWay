#!/bin/bash

python3 train.py --i 0 --name deep_way --clear_file
python3 train.py --i 1 --name deep_way
python3 train.py --i 2 --name deep_way

python3 train.py --i 0 --name deep_way --curved
python3 train.py --i 1 --name deep_way --curved
python3 train.py --i 2 --name deep_way --curved

python3 train.py --i 0 --name cluster_way
python3 train.py --i 1 --name cluster_way
python3 train.py --i 2 --name cluster_way

python3 train.py --i 0 --name cluster_way --curved
python3 train.py --i 1 --name cluster_way --curved
python3 train.py --i 2 --name cluster_way --curved