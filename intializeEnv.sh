#!/bin/bash

# Run this script in the users root folder i.e. home/user


# Pull git repos

git clone https://github.com/Slaysenfite/ddsm_lr.git
git clone https://github.com/Slaysenfite/masters-tensorflow2-model.git

# Make virtual environment and install requirements

mkvirtualenv wesselsenv -p python3
workon wesselsenv

cd masters-tensorflow2-model/
pip install -r requirements.txt

# Create metadata file for the DDSM

git checkout instance/uber

python python_src/configurations/DdsmMetadataGenerator.py



