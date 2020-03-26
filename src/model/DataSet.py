from enum import Enum

# Dataset paths

ROOT_DIRECTORY = '/home/slaysenfite/dev/data'
PATH_TO_DDSM = '/ddsm_lr'
PATH_TO_MIAS = '/mias'
PATH_TO_INBREAST = '/inbreast'

# Dataset labels

ddsm_label_map = {'B': 0, 'M': 1, 'N': 2}
ddsm_class_names = ['Benign', 'Malignant', 'Normal']


class DataSet:
    def __init__(self, name, root_path, metadata_path, class_label_index, label_map):
        self.name = name
        self.root_path = root_path
        self.metadata_path = metadata_path
        self.class_label_index = class_label_index
        self.label_map = label_map


class DataSetNames(Enum):
    DDSM = "Digital Database for Screening Mammography"
    InBreast = "IN"


def create_ddsm_dataset():
    return DataSet(
        DataSetNames.DDSM,
        ROOT_DIRECTORY + PATH_TO_DDSM,
        ROOT_DIRECTORY + PATH_TO_DDSM + '/ddsm.csv',
        1,
        ddsm_label_map
    )


DDSM_DataSet = create_ddsm_dataset()
