import os

from utils.DicomUtils import decompress_and_convert_dicom

RELATIVE_DATA_SET_PATH = '/media/slaysenfite/Seagate Expansion Drive/manifest-1605042674814/Breast-Cancer-Screening-DBT'
home = os.path.expanduser("~")

# path_to_data_folder = home + RELATIVE_DATA_SET_PATH


def clease_dataset():
    count = 0
    for dirName, subdirList, fileList in os.walk('/media/slaysenfite/Seagate Expansion Drive/manifest-1605042674814/Breast-Cancer-Screening-DBT'):
        for fname in fileList:
            if '1-1.dcm' in fname:
                count = count + 1 # decompress_and_convert_dicom(dirName + '/' + fname)
    print(count)



def append_to_csv(path, string):
    f = open(path, 'a+')
    f.write(string + '\n')
    print('appended line: ' + string)


clease_dataset()
