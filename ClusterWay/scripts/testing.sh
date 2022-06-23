#!/bin/bash

python3 test.py --i 0 --name deep_way --clear_file true
python3 test.py --i 1 --name deep_way
python3 test.py --i 2 --name deep_way

python3 test.py --i 0 --name deep_way --curved true
python3 test.py --i 1 --name deep_way --curved true
python3 test.py --i 2 --name deep_way --curved true

python3 test.py --i 0 --name cluster_way
python3 test.py --i 1 --name cluster_way
python3 test.py --i 2 --name cluster_way

python3 test.py --i 0 --name cluster_way --curved true
python3 test.py --i 1 --name cluster_way --curved true
python3 test.py --i 2 --name cluster_way --curved true