import os

from pandas import read_csv

RELATIVE_DATA_SET_PATH = '/data/BCS-DBT-PNG'
STUPID_ASS_IMAGE_NAME = "1-1-decompresse.png"

home = os.path.expanduser("~")

path_to_data_folder = home + RELATIVE_DATA_SET_PATH


def generate_bcs_dbt_metadata_file():
    os.remove(path_to_data_folder + '/bcs-dbt-metadata.csv')
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


def append_to_csv(path, string):
    f = open(path, 'a+')
    f.write(string + '\n')
    print('appended line: ' + string)

generate_bcs_dbt_metadata_file()