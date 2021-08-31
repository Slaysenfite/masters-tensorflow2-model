#!/bin/bash

# Run this script in the users root folder i.e. home/user

cd $HOME

# Pull git repos

git clone https://github.com/Slaysenfite/masters-tensorflow2-model.git

# Create data directory

mkdir data
cd data

git clone https://github.com/Slaysenfite/CBIS-DDSM-PNG.git
git clone https://github.com/Slaysenfite/BCS-DBT-PNG

cd $HOME

# Make virtual environment and install requirements

mkvirtualenv wesselsenv -p python3
workon wesselsenv

cd $HOME/masters-tensorflow2-model/
pip install -r requirements.txt

# Create metadata file for the Datasets

git checkout instance/uber

python python_src/configurations/CbisDdsmMetadataGenerator.py
python python_src/configurations/BcsDbtUtils.py



