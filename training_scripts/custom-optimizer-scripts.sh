#!/bin/bash

echo "Running standard pipelines"

echo "Hybrid Model"
python python_src/TestHybridMain.py

echo "PSO Model"
python python_src/TestPsoMain.py


