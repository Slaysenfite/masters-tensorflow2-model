import os
import time
from datetime import timedelta

from utils.DicomUtils import decompress_and_convert_dicom

RELATIVE_DATA_SET_PATH = '/home/slaysenfite/Desktop/fg'
home = os.path.expanduser("~")

# path_to_data_folder = home + RELATIVE_DATA_SET_PATH


def clease_dataset():
    for dirName, subdirList, fileList in os.walk(RELATIVE_DATA_SET_PATH):
        for fname in fileList:
            if '1-1.dcm' in fname:
                start_time = time.time()
                decompress_and_convert_dicom(dirName + '/' + fname)
                print(timedelta(seconds=(time.time() - start_time)))



def append_to_csv(path, string):
    f = open(path, 'a+')
    f.write(string + '\n')
    print('appended line: ' + string)


clease_dataset()
