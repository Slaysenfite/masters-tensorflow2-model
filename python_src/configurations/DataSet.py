from enum import Enum
from os.path import expanduser

import numpy as np
import pandas as pd
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.keras.utils.np_utils import to_categorical

home = expanduser("~")

# Dataset paths

ROOT_DIRECTORY = home + '/data'
PATH_TO_DDSM = '/ddsm_lr_sample'
PATH_TO_MIAS = '/mias'
PATH_TO_CBIS_DDSM = '/CBIS-DDSM_CLASSIC_PNG'


# Dataset labels

two_class_label_map = {'P': 0, 'N': 1}
two_class_names = ['Positive', 'Negative']

three_class_label_map = {'B': 0, 'M': 1, 'N': 2}
three_class_names = ['Benign', 'Malignant', 'Normal']


class DataSetNames(Enum):
    DDSM = 'Digital Database for Screening Mammography'
    CBIS_DDSM = 'Curated Breast Imaging Subset of DDSM'
    MIAS = 'Mammographic Image Analysis Homepage'
    InBreast = 'IN'
    MNIST = 'Modified National Institute of Standards and Technology'

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

    def split_data_set(self, data, labels):
        return train_test_split(data, labels, test_size=0.3, train_size=0.7,
                                random_state=42)

    def get_dataset_labels(self, train_y, test_y):
        if self.is_multiclass:
            print('[INFO] Configure for multiclass classification')
            lb = LabelBinarizer()
            train_y = lb.fit_transform(train_y)
            test_y = lb.transform(test_y)
            loss = 'categorical_crossentropy'
        else:
            print('[INFO] Configure for binary classification')
            train_y = to_categorical(train_y)
            test_y = to_categorical(test_y)
            loss = 'binary_crossentropy'
        return loss, train_y, test_y

    def get_image_paths(self):
        return list(paths.list_images(self.root_path))

    def get_ground_truth_lables(label_index):
        return


class MultiPartDataset(DataSet):

    def get_image_paths(self):
        df_paths = pd.read_csv(self.train_metadata_path)
        df_paths.append(pd.read_csv(self.test_metadata_path))
        return df_paths['image'].to_list()

    def get_image_metadata(self):
        df_train_metadata = pd.read_csv(self.train_metadata_path)[['image', 'label']]
        df_test_metadata = pd.read_csv(self.test_metadata_path)[['image', 'label']]
        total_metadata = df_train_metadata
        total_metadata.append(df_test_metadata)
        return np.array(total_metadata)

    def split_data_set(self, data, labels):
        return train_test_split(data, labels, test_size=0.25, train_size=0.75, random_state=None, shuffle=False)

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
    return MultiPartDataset(
        DataSetNames.CBIS_DDSM.name,
        ROOT_DIRECTORY + PATH_TO_CBIS_DDSM,
        ROOT_DIRECTORY + PATH_TO_CBIS_DDSM + '/train_cbis-ddsm.csv',
        ROOT_DIRECTORY + PATH_TO_CBIS_DDSM + '/test_cbis-ddsm.csv',
        1,
        two_class_label_map,
        two_class_names,
        False
    )


def create_cbis_ddsm_five_class_dataset_singleton():
    return MultiPartDataset(
        DataSetNames.CBIS_DDSM.name,
        ROOT_DIRECTORY + PATH_TO_CBIS_DDSM,
        ROOT_DIRECTORY + PATH_TO_CBIS_DDSM + '/train_cbis-ddsm_five.csv',
        ROOT_DIRECTORY + PATH_TO_CBIS_DDSM + '/test_cbis-ddsm_five.csv',
        1,
        {0:0, 1:1, 2:2, 3:3, 4:4, 5:5},
        ['0', '1', '2', '3', '4', '5'],
        True
    )


def create_mnist_dataset_singleton():
    return DataSet(
        DataSetNames.MNIST.name,
        None,
        None,
        None,
        0,
        {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9},
        ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'],
        True
    )


ddsm_data_set = create_ddsm_three_class_dataset_singleton()
binary_ddsm_data_set = create_ddsm_two_class_dataset_singleton()
mias_data_set = create_mias_dataset_singleton()
cbis_ddsm_data_set = create_cbis_ddsm_dataset_singleton()
cbis_ddsm_five_data_set = create_cbis_ddsm_five_class_dataset_singleton()
mnist_data_set = create_mnist_dataset_singleton()
