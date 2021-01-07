from argparse import ArgumentParser

from configurations.TrainingConfig import output_dir
from metrics.MetricResults import MetricResult


def generate_script_report(H, test_y, predictions, data_set, hyperparameters, model_name, metadata_string=''):
    metric_result = MetricResult(H, test_y, predictions, data_set)
    with open(output_dir + model_name + '_' + data_set.name + '_' + metadata_string + '_result_report.txt',
              'w+') as text_file:
        text_file.write(
            hyperparameters.report_hyperparameters() + metric_result.report_result()
        )
    print(metric_result.report_result())


def read_cmd_line_args(dataset, hyperparameters, image_dims):
    parser = ArgumentParser(description='Enter the path to the data set')
    parser.add_argument('--data_set_path', type=str)
    parser.add_argument('--set_image_dims', type=int)
    parser.add_argument('--set_num_epochs', type=int)
    args = parser.parse_args()
    if (args.data_set_path != None):
        dataset.root_path = args.data_set_path
        dataset.metadata_path = args.data_set_path + '/ddsm.csv'
    if (args.set_image_dims != None):
        temp_dims = (args.set_image_dims, args.set_image_dims, image_dims[2])
        image_dims = temp_dims
    if (args.set_num_epochs != None):
        hyperparameters.epochs = args.set_num_epochs
