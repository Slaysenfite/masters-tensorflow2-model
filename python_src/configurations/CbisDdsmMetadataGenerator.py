import os
from os.path import expanduser

import pandas as pd

home = expanduser("~")

image_file_tuple = ('image file path', '1-1', 'full_image')
cropped_image_tuple = ('cropped image file path', '1-1', 'cropped_image')
roi_mask_tuple = ('ROI mask file path', '1-2', 'roi_image')

RELATIVE_DATA_SET_PATH = '/data/CBIS-DDSM-PNG/'


def append_to_ddsm_csv(path, string):
    f = open(path, 'a+')
    f.write(string + '\n')
    print('appended line: ' + string)


def generate_cbis_ddsm_metadata_file(cbis_ddsm_csv, metadata_csv_path, extraction_tuple=image_file_tuple):
    df_csv = pd.read_csv(home + RELATIVE_DATA_SET_PATH + cbis_ddsm_csv)
    path_to_data_folder = home + RELATIVE_DATA_SET_PATH
    for index, row in df_csv.iterrows():
        image_path = row[extraction_tuple[0]].strip('\t\n\r')
        append_row(image_path, metadata_csv_path, path_to_data_folder, row)


def append_row(image_path, metadata_csv_path, path_to_data_folder, row):
    if 'BENIGN' in row['pathology']:
        data = path_to_data_folder + image_path + ',' + 'N'
        append_to_ddsm_csv(path_to_data_folder + metadata_csv_path, data)
    if 'MALIGNANT' in row['pathology']:
        data = path_to_data_folder + image_path + ',' + 'P'
        append_to_ddsm_csv(path_to_data_folder + metadata_csv_path, data)


def generate_cbis_ddsm_five_metadata_file(cbis_ddsm_csv, metadata_csv_path, extraction_tuple=image_file_tuple):
    df_csv = pd.read_csv(home + RELATIVE_DATA_SET_PATH + cbis_ddsm_csv)
    path_to_data_folder = home + RELATIVE_DATA_SET_PATH
    for index, row in df_csv.iterrows():
        image_path = row[extraction_tuple[0]].strip('\t\n\r')
        if extraction_tuple[1] in image_path:
            append_fives_row(image_path, metadata_csv_path, path_to_data_folder, row)


def append_fives_row(image_path, metadata_csv_path, path_to_data_folder, row):
    if 0 == row['assessment']:
        data = path_to_data_folder + image_path + ',' + '0'
        append_to_ddsm_csv(path_to_data_folder + metadata_csv_path, data)
    if 1 == row['assessment']:
        data = path_to_data_folder + image_path + ',' + '1'
        append_to_ddsm_csv(path_to_data_folder + metadata_csv_path, data)
    if 2 == row['assessment']:
        data = path_to_data_folder + image_path + ',' + '2'
        append_to_ddsm_csv(path_to_data_folder + metadata_csv_path, data)
    if 3 == row['assessment']:
        data = path_to_data_folder + image_path + ',' + '3'
        append_to_ddsm_csv(path_to_data_folder + metadata_csv_path, data)
    if 4 == row['assessment']:
        data = path_to_data_folder + image_path + ',' + '4'
        append_to_ddsm_csv(path_to_data_folder + metadata_csv_path, data)
    if 5 == row['assessment']:
        data = path_to_data_folder + image_path + ',' + '5'
        append_to_ddsm_csv(path_to_data_folder + metadata_csv_path, data)


def create_cbis_metadata_file(tuple=cropped_image_tuple):
    os.remove(home + RELATIVE_DATA_SET_PATH + tuple[2] + '_train_cbis-ddsm.csv')
    os.remove(home + RELATIVE_DATA_SET_PATH + tuple[2] + '_test_cbis-ddsm.csv')
    append_to_ddsm_csv(home + RELATIVE_DATA_SET_PATH + tuple[2] + '_train_cbis-ddsm.csv', 'image,label')
    append_to_ddsm_csv(home + RELATIVE_DATA_SET_PATH + tuple[2] + '_test_cbis-ddsm.csv', 'image,label')
    generate_cbis_ddsm_metadata_file('calc_case_description_train_set.csv', tuple[2] + '_train_cbis-ddsm.csv', tuple)
    generate_cbis_ddsm_metadata_file('mass_case_description_train_set.csv', tuple[2] + '_train_cbis-ddsm.csv', tuple)
    generate_cbis_ddsm_metadata_file('calc_case_description_test_set.csv', tuple[2] + '_test_cbis-ddsm.csv', tuple)
    generate_cbis_ddsm_metadata_file('mass_case_description_test_set.csv', tuple[2] + '_test_cbis-ddsm.csv', tuple)


def create_cbis_five_metadata_file(tuple=cropped_image_tuple):
    os.remove(home + RELATIVE_DATA_SET_PATH + tuple[2] + '_train_cbis-ddsm_five.csv')
    os.remove(home + RELATIVE_DATA_SET_PATH + tuple[2] + '_test_cbis-ddsm_five.csv')
    append_to_ddsm_csv(home + RELATIVE_DATA_SET_PATH + tuple[2] + '_train_cbis-ddsm_five.csv', 'image,label')
    append_to_ddsm_csv(home + RELATIVE_DATA_SET_PATH + tuple[2] + '_test_cbis-ddsm_five.csv', 'image,label')
    generate_cbis_ddsm_five_metadata_file('calc_case_description_train_set.csv', tuple[2] + '_train_cbis-ddsm_five.csv', tuple)
    generate_cbis_ddsm_five_metadata_file('mass_case_description_train_set.csv', tuple[2] + '_train_cbis-ddsm_five.csv', tuple)
    generate_cbis_ddsm_five_metadata_file('calc_case_description_test_set.csv', tuple[2] + '_test_cbis-ddsm_five.csv', tuple)
    generate_cbis_ddsm_five_metadata_file('mass_case_description_test_set.csv', tuple[2] + '_test_cbis-ddsm_five.csv', tuple)


create_cbis_metadata_file(image_file_tuple)
create_cbis_metadata_file(roi_mask_tuple)
create_cbis_metadata_file(cropped_image_tuple)
