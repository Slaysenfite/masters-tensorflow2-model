import os


def gen_ddsm_metadata(rootDir):
    csv_path = rootDir + '/ddsm.csv'
    append_to_ddsm_csv(csv_path, 'image,label')
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            if 'png' in fname:
                if 'benign' in dirName:
                    append_to_ddsm_csv(csv_path, fname + ',B')
                elif 'cancer' in dirName:
                    append_to_ddsm_csv(csv_path, fname + ',M')
                elif 'normal' in dirName:
                    append_to_ddsm_csv(csv_path, fname + ',N')


def append_to_ddsm_csv(path, string):
    f = open(path, 'a+')
    f.write(string + '\n')
    print('appended line: ' + string)