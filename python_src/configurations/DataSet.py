from enum import Enum
from os.path import expanduser

import numpy as np
import pandas as pd

home = expanduser("~")

# Dataset paths

ROOT_DIRECTORY = home + '/dev/data'
PATH_TO_DDSM = '/ddsm_lr_sample'
PATH_TO_MIAS = '/mias'
PATH_TO_CBIS_DDSM = '/CBIS-DDSM_CLASSIC_PNG'


# Dataset labels

two_class_label_map = {'P': 0, 'N': 1}
two_class_names = ['Positive', 'Negative']

three_class_label_map = {'B': 0, 'M': 1, 'N': 2}
three_class_names = ['Benign', 'Malignant', 'Normal']


class DataSet:
    def __init__(self, name, root_path, train_metadata_path, test_metadata_path, class_label_index, label_map,
                 class_names, is_multiclass):
        self.name = name
        self.root_path = root_path
        self.train_metadata_path = train_metadata_path
        self.test_metadata_path = test_metadata_path
        self.class_label_index = class_label_index
        self.label_map = label_map
        self.class_names = class_names
        self.is_multiclass = is_multiclass

    def get_image_metadata(self):
        df_images = pd.read_csv(self.train_metadata_path)
        return np.array(df_images)


class DataSetNames(Enum):
    DDSM = 'Digital Database for Screening Mammography'
    CBIS_DDSM = 'Curated Breast Imaging Subset of DDSM'
    MIAS = 'Mammographic Image Analysis Homepage'
    InBreast = 'IN'


def create_ddsm_three_class_dataset_singleton():
    return DataSet(
        DataSetNames.DDSM.name,
        ROOT_DIRECTORY + PATH_TO_DDSM,
        ROOT_DIRECTORY + PATH_TO_DDSM + '/ddsm.csv',
        None,
        1,
        three_class_label_map,
        three_class_names,
        True
    )

def create_ddsm_two_class_dataset_singleton():
    return DataSet(
        DataSetNames.DDSM.name,
        ROOT_DIRECTORY + PATH_TO_DDSM,
        ROOT_DIRECTORY + PATH_TO_DDSM + '/binary_ddsm.csv',
        None,
        1,
        two_class_label_map,
        two_class_names,
        False
    )


def create_mias_dataset_singleton():
    return DataSet(
        DataSetNames.MIAS.name,
        ROOT_DIRECTORY + PATH_TO_MIAS,
        ROOT_DIRECTORY + PATH_TO_MIAS + '/info.csv',
        None,
        3,
        three_class_label_map,
        three_class_names,
        True
    )


def create_cbis_ddsm_dataset_singleton():
    return DataSet(
        DataSetNames.CBIS_DDSM.name,
        ROOT_DIRECTORY + PATH_TO_CBIS_DDSM,
        ROOT_DIRECTORY + PATH_TO_CBIS_DDSM + '/train_cbis-ddsm.csv',
        ROOT_DIRECTORY + PATH_TO_CBIS_DDSM + '/test_cbis-ddsm.csv',
        1,
        two_class_label_map,
        two_class_names,
        False
    )

ddsm_data_set = create_ddsm_three_class_dataset_singleton()
binary_ddsm_data_set = create_ddsm_two_class_dataset_singleton()
mias_data_set = create_mias_dataset_singleton()
cbis_ddsm_data_set = create_cbis_ddsm_dataset_singleton()

