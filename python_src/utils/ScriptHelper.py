import os
from metrics.MetricResults import MetricResult
from configurations.GConstants import output_dir
from argparse import ArgumentParser




def generate_script_report(H, test_y, predictions, data_set, hyperparameters, model_name):
    metric_result = MetricResult(H, test_y, predictions, data_set)
    with open(output_dir + model_name + '_' + data_set.name + '_result_report.txt', 'w+') as text_file:
        text_file.write(
            hyperparameters.report_hyperparameters() + metric_result.report_result()
        )
    print(metric_result.report_result())


def read_cmd_line_args(dataset):
    parser = ArgumentParser(description='Enter the path to the data set')
    parser.add_argument('--data_set_path', type=str)
    args = parser.parse_args()
    if (args.data_set_path != None):
        dataset.root_path = args.data_set_path
        dataset.metadata_path = args.data_set_path + '/ddsm.csv'



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


