import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
from tensorflow_datasets.image_classification import CuratedBreastImagingDDSM

ddsm, ddsm_info = tfds.load('curated_breast_imaging_ddsm', split='train', with_info=True)
fig = tfds.show_examples(ddsm, ddsm_info)
