#!/bin/bash

echo "Running standard pipelines"

echo "Inception V3"
python python_src/GooLeNetMain.py

echo "U-Net"
python python_src/UNetMain.py

echo "ResNet"
python python_src/ResNetMain.py
