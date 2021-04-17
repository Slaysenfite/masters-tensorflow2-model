import os
import time
from datetime import timedelta

from pandas import read_csv

from utils.DicomUtils import decompress_and_convert_dicom

RELATIVE_DATA_SET_PATH = '/data/BCS-DBT-PNG'
STUPID_ASS_IMAGE_NAME = "1-1-decompresse.png"

home = os.path.expanduser("~")

path_to_data_folder = home + RELATIVE_DATA_SET_PATH


def generate_bcs_dbt_metadata_file():
    append_to_csv(path_to_data_folder + '/bcs-dbt-metadata.csv', 'image,label')
    df_paths = read_csv(path_to_data_folder + '/BCS-DBT file-paths-train.csv')
    df_labels = read_csv(path_to_data_folder + '/BCS-DBT labels-train.csv')

    for index, row in df_paths.iterrows():
        original_path = row['descriptive_path']
        new_path = original_path.replace('1-1.dcm', STUPID_ASS_IMAGE_NAME).replace('Breast-Cancer-Screening-DBT',
                                                                                   path_to_data_folder)
        label_row = df_labels.iloc[index]
        if label_row['StudyUID'] == row['StudyUID'] and label_row['View'] == row['View']:
            normal = label_row['Normal']
            if normal == 1:
                label = 'N'
            else:
                label = 'P'
        append_to_csv(path_to_data_folder + '/bcs-dbt-metadata.csv', new_path + ',' + label)


def cleanse_dataset():
    for dirName, subdirList, fileList in os.walk(RELATIVE_DATA_SET_PATH):
        for fname in fileList:
            if '1-1.dcm' in fname or '1-2.dcm' in fname:
                start_time = time.time()
                decompress_and_convert_dicom(dirName, fname)
                print(timedelta(seconds=(time.time() - start_time)))


def append_to_csv(path, string):
    f = open(path, 'a+')
    f.write(string + '\n')
    print('appended line: ' + string)


generate_bcs_dbt_metadata_file()
# cleanse_dataset()
