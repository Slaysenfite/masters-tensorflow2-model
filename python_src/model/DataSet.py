from enum import Enum

import numpy as np
import pandas as pd

# Dataset paths

ROOT_DIRECTORY = '/home/stedeweasles/dev/data/'
PATH_TO_DDSM = '/ddsm_lr'
PATH_TO_MIAS = '/mias'
PATH_TO_INBREAST = '/inbreast'

# Dataset labels

ddsm_label_map = {'B': 0, 'M': 1, 'N': 2}
ddsm_class_names = ['Benign', 'Malignant', 'Normal']


class DataSet:
    def __init__(self, name, root_path, metadata_path, class_label_index, label_map, class_names):
        self.name = name
        self.root_path = root_path
        self.metadata_path = metadata_path
        self.class_label_index = class_label_index
        self.label_map = label_map
        self.class_names = class_names

    def get_image_metadata(self):
        df_images = pd.read_csv(self.metadata_path)
        return np.array(df_images)


class DataSetNames(Enum):
    DDSM = 'Digital Database for Screening Mammography'
    MIAS = 'Mammographic Image Analysis Homepage'
    InBreast = 'IN'


def create_ddsm_dataset_singleton():
    return DataSet(
        DataSetNames.DDSM.name,
        ROOT_DIRECTORY + PATH_TO_DDSM,
        ROOT_DIRECTORY + PATH_TO_DDSM + '/ddsm.csv',
        1,
        ddsm_label_map,
        ddsm_class_names
    )


def create_mias_dataset_singleton():
    return DataSet(
        DataSetNames.MIAS.name,
        ROOT_DIRECTORY + PATH_TO_MIAS,
        ROOT_DIRECTORY + PATH_TO_MIAS + '/info.csv',
        3,
        ddsm_label_map,
        ddsm_class_names
    )

ddsm_data_set = create_ddsm_dataset_singleton()
mias_data_set = create_mias_dataset_singleton()
