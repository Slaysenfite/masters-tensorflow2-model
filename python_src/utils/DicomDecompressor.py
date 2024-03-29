# coding=utf-8
# -----------------------------------------------------
# decompressDICOM.py
#
# Created by:   Michael Kuczynski
# Created on:   2018.10.19
#
# Description: Decompresses compressed DICOM images using pydicom.
# -----------------------------------------------------
#
# Requirements:
#   -Python 2.7, 3.4 or later
#   -pydicom, GDCM, Pillow, argparse, shutil, errno
#
# Usage:
#   decompressDICOM.py DICOM_FOLDER
# -----------------------------------------------------

import pydicom
import os
import argparse
import errno
import shutil

# DICOM Decompression:

# 1. Read in the input DICOM directory


inputDirectory = '/home/slaysenfite/Desktop/cunt/DBT-P00095/01-01-2000-DBT-S05144-Standard Screening - TomoHD-83995/5226.000000-36504' + '/'

print("Starting decompression...\n")

# 2. Create a new folder for the decompressed DICOM files
savePath = inputDirectory + "/DECOMPRESSED/"

try:
    os.mkdir(savePath)
except OSError as e:
    if e.errno != errno.EEXIST:  # File already exists error
        raise

# 3. Loop through the entire input folder
#       First decompress the file
#       Next, get the tag of each DICOM image to analyze the series description
#       Finally, place the decompressed image into the correct series folder
for DICOMfile in os.listdir(inputDirectory):
    # Get the next item in the directory
    ogFilename = os.fsdecode(DICOMfile)

    # We need to skip any directories and loop over files only
    if os.path.isdir(inputDirectory + ogFilename):
        continue

    # Decompress the file
    ds = pydicom.dcmread(inputDirectory + ogFilename)
    ds.decompress()

    # Save the file to the correct series folder
    saveFile = savePath + ogFilename + ".dcm"

    try:
        ds.save_as(saveFile)
    except OSError as e:
        if e.errno != errno.ENOENT:  # No such file or directory error
            print("ERROR: No such file or directory named " + saveFile)
            raise

    # Get the file's tag and parse out the series description
    # Series description is located at [0x0008, 0x103e] in the tag and can be one of the following:
    #   1. Scout
    #   2. Bone Plus
    #   3. Standard
    #   4. No Calibration Phantom (DFOV)
    #   5. Default (series # 601)
    #   6. Default (series # 602)
    #   7. Dose Report
    #   8. Localizers
    tag = pydicom.read_file(saveFile)
    seriesDescription = str(tag[0x0008, 0x103e].value).upper()
    seriesNumber = tag[0x0020, 0x0011].value

    seriesFilePath = savePath + seriesDescription + "\\"

    # Possible race-condition with creating directories like this...
    # TO-DO: Fix...
    if not os.path.exists(seriesFilePath):
        os.makedirs(seriesFilePath)

    shutil.move(saveFile, seriesFilePath + ogFilename + ".dcm")

print("\nDONE!")