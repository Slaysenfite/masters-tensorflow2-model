import errno
import os

import cv2
import numpy as np
import pydicom


# DICOM Decompression:

def decompress_and_convert_dicom(dicomFile):

    ogFilename = os.fsdecode(dicomFile)

    # Decompress the file
    print('Processing: ' + ogFilename)
    ds = pydicom.dcmread(ogFilename)
    ds.decompress()

    # Save the file to the correct series folder
    saveFile = f'{ogFilename.strip(".dcm")}-{"decompressed"}.dcm'

    try:
        ds.save_as(saveFile)
    except OSError as e:
        if e.errno != errno.ENOENT:  # No such file or directory error
            print('ERROR: No such file or directory named ' + saveFile)
            raise

    convert_to_png(saveFile)

    os.remove(saveFile)

    os.remove(ogFilename)


def convert_to_png(filename):
    ds = pydicom.read_file(filename)

    shape = ds.pixel_array.shape

    images = ds.pixel_array

    for i in range(len(images)):
        # Convert to float to avoid overflow or underflow losses.
        image_2d = images[i].astype(float)

        # Rescaling grey scale between 0-255
        image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0

        # Convert to uint
        image_2d_scaled = np.uint8(image_2d_scaled)

        # Write the PNG file
        cv2.imwrite(f'{filename.strip(".dcm")}-{i}.png', image_2d_scaled)
