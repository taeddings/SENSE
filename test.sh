#!/bin/bash
cd /data/data/com.termux/files/home/project/SENSE/SENSE
export PYTHONPATH=/data/data/com.termux/files/home/project/SENSE/SENSE:$PYTHONPATH
python run_tests.py
