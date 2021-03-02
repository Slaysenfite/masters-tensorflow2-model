import cv2
import numpy as np
import pydicom as pydicom


def convert_to_png(file):
    ds = pydicom.read_file(file)

    shape = ds.pixel_array.shape

    # Convert to float to avoid overflow or underflow losses.
    image_2d = ds.pixel_array.astype(float)[0]

    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0

    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)

    # image = cv2.resize(image_2d_scaled, (shape[1], shape[2]))

    # Write the PNG file
    cv2.imwrite(f'{file.strip(".dcm")}.png', image_2d_scaled)


convert_to_png(
    '/home/slaysenfite/Desktop/cunt/DBT-P00036/01-01-2000-DBT-S03354-MAMMO screening digital bilateral-14628/12277.000000-24120/DECOMPRESSED/1-1.dcm')
