#!/bin/bash

echo "Running standard pipelines"

echo "Inception V3"
GooLeNetMain.py

echo "U-Net"
UNetMain.py

echo "ResNet"
ResNetMain.py
