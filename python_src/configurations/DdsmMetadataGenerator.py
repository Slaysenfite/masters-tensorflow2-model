import os

from configurations.DataSet import ddsm_data_set


def gen_ddsm_metadata(rootDir):
    csv_path = rootDir + '/ddsm.csv'
    append_to_ddsm_csv(csv_path, 'image,label')
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            if 'png' in fname:
                if 'normal' in dirName:
                    append_to_ddsm_csv(csv_path, fname + ',N')
                elif 'benign' in dirName and check_for_overlay(fname, dirName) is True:
                    append_to_ddsm_csv(csv_path, fname + ',B')
                elif 'cancer' in dirName and check_for_overlay(fname, dirName) is True:
                    append_to_ddsm_csv(csv_path, fname + ',M')
                else:
                    append_to_ddsm_csv(csv_path, fname + ',N')

def gen_binary_classification_ddsm_metadata(rootDir):
    csv_path = rootDir + '/binary_ddsm.csv'
    append_to_ddsm_csv(csv_path, 'image,label')
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
                if 'normal' in dirName:
                    append_to_ddsm_csv(csv_path, fname + ',N')
                elif 'benign' in dirName and check_for_overlay(fname, dirName) is True:
                    append_to_ddsm_csv(csv_path, fname + ',P')
                elif 'cancer' in dirName and check_for_overlay(fname, dirName) is True:
                    append_to_ddsm_csv(csv_path, fname + ',P')
                else:
                    append_to_ddsm_csv(csv_path, fname + ',N')


def append_to_ddsm_csv(path, string):
    f = open(path, 'a+')
    f.write(string + '\n')
    print('appended line: ' + string)


def check_for_overlay(filename, path):
    name = filename[0:-4]
    parent_path = path[0:-9]
    for fileList in os.walk(parent_path):
        for fname in fileList[2]:
            if (name+'.OVERLAY') in fname:
                return True
    return False


def retrieve_overlay_data(filename):
    return None


def curate_cbis_ddsm_folder(path_to_dir, ext_sequence, strip_sequence):
    for fileList in os.walk(path_to_dir):
        for fname in fileList[2]:
            if ext_sequence in fname:
                os.remove(os.path.join(fileList[0], fname))
            if strip_sequence in fname:
                new_fname = fname.replace(strip_sequence, '')
                os.rename(os.path.join(fileList[0], fname), os.path.join(fileList[0], new_fname))


gen_binary_classification_ddsm_metadata(ddsm_data_set.root_path)
