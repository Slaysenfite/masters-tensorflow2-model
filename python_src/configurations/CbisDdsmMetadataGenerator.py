from os.path import expanduser

import pandas as pd

home = expanduser("~")

IMAGE_COL_NAME = 'image file path'
RELATIVE_DATA_SET_PATH = '/data/CBIS-DDSM_CLASSIC_PNG/'


def append_to_ddsm_csv(path, string):
    f = open(path, 'a+')
    f.write(string + '\n')
    print('appended line: ' + string)


def generate_cbis_ddsm_metadata_file(cbis_ddsm_csv, metadata_csv_path):
    df_csv = pd.read_csv(home + RELATIVE_DATA_SET_PATH + cbis_ddsm_csv)
    path_to_data_folder = home + RELATIVE_DATA_SET_PATH
    for index, row in df_csv.iterrows():
        image_path = row[IMAGE_COL_NAME].strip('\t\n\r')
        if 'BENIGN' in row['pathology']:
            data = path_to_data_folder + image_path + ',' + 'N'
            append_to_ddsm_csv(path_to_data_folder + metadata_csv_path, data)
        if 'MALIGNANT' in row['pathology']:
            data = path_to_data_folder + image_path + ',' + 'P'
            append_to_ddsm_csv(path_to_data_folder + metadata_csv_path, data)

append_to_ddsm_csv(home + RELATIVE_DATA_SET_PATH + 'train_cbis-ddsm.csv', 'image,label')
append_to_ddsm_csv(home + RELATIVE_DATA_SET_PATH + 'test_cbis-ddsm.csv', 'image,label')
generate_cbis_ddsm_metadata_file('calc_case_description_train_set.csv', 'train_cbis-ddsm.csv')
generate_cbis_ddsm_metadata_file('mass_case_description_train_set.csv', 'train_cbis-ddsm.csv')
generate_cbis_ddsm_metadata_file('calc_case_description_test_set.csv', 'test_cbis-ddsm.csv')
generate_cbis_ddsm_metadata_file('mass_case_description_test_set.csv', 'test_cbis-ddsm.csv')

# df_csv = pd.read_csv(home + RELATIVE_DATA_SET_PATH + 'train_cbis-ddsm.csv')
#
# print(df_csv)


