import os


def clean_dir(filePath):
    if os.path.exists(filePath):
        os.remove(filePath)
    else:
        print("File {} does not exist".format(filePath))
