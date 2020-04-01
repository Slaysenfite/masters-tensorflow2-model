import gc

import cv2
import numpy as np
import pandas as pd
from imutils import paths


def load_images(data, labels, arr_dataset, dataset, image_dimensions=(128, 128, 3)):
    print("[INFO] loading images...")
    """Add these to dataset object"""
    df_images = pd.read_csv(dataset.metadata_path)
    arr_images = np.array(df_images)

    # grab the image paths and randomly shuffle them
    image_paths = paths.list_images(dataset.metadata_path)

    i = 0
    # loop over the input images
    for imagePath in image_paths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (image_dimensions[1], image_dimensions[0]))
        data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = dataset.label_map[arr_dataset[i][dataset.class_label_index]]

        labels.append(label)
        i += 1

    gc.collect()

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    print("[INFO] Data shape: " + str(data.shape))
    print("[INFO] Label shape: " + str(labels.shape))

    return data, labels
