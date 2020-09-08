#!/bin/bash

echo "Installing requirements..."

pip install -r requirements.txt

echo "Running standard pipelines"

echo "VggNet19"
python python_src/VggNetMain.py

echo "Inception V3"
python python_src/GooLeNetMain.py.py

echo "U-Net"
python python_src/UNetMain.py

echo "ResNet"
python python_src/ResNetMain.py
