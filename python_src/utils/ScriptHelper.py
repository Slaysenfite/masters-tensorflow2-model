from argparse import ArgumentParser

from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD

from configurations.DataSet import cbis_ddsm_data_set, bcs_data_set, cbis_seg_data_set
from configurations.TrainingConfig import output_dir
from metrics.MetricResults import MetricResult
from metrics.PerformanceMetrics import PerformanceMetrics


def generate_script_report(H, model, test_x, test_y, predictions, time_taken, data_set, hyperparameters, model_name,
                           metadata_string=''):
    metric_result = MetricResult(model, H, test_x, test_y, predictions, data_set)
    performance_metrics = PerformanceMetrics(time_taken=time_taken)
    with open(
            output_dir + hyperparameters.experiment_id + '_' + model_name + '_' + data_set.name + '_' + metadata_string
            + '_result_report.txt',
              'w+') as text_file:
        text_file.write(
            hyperparameters.report_hyperparameters()
            + performance_metrics.report_metrics()
            + metric_result.report_result()
        )
    print(metric_result.report_result())


def read_cmd_line_args(hyperparameters, dataset):
    parser = ArgumentParser()
    parser.add_argument('--meta_heuristic', type=str)
    parser.add_argument('--meta_heuristic_order', type=str)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--id', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--augmentation', type=str)
    parser.add_argument('--preloaded_weights', type=str)
    parser.add_argument('--tf_fit', type=str)
    args = parser.parse_args()

    if args.id is not None:
        hyperparameters.experiment_id = args.id

    if args.meta_heuristic is not None:
        hyperparameters.meta_heuristic = args.meta_heuristic

    if args.meta_heuristic_order is not None and args.meta_heuristic_order == 'first':
        hyperparameters.meta_heuristic_order = 'first'
    elif args.meta_heuristic_order is not None and args.meta_heuristic_order == 'last':
        hyperparameters.meta_heuristic_order = 'last'

    if args.optimizer is not None and args.optimizer == 'adam':
        hyperparameters.learning_optimization = 'Adam'
        opt = Adam(learning_rate=hyperparameters.adam_lr, decay=True)
    else:
        hyperparameters.learning_optimization = 'Stochastic Gradient Descent'
        opt = SGD(lr=hyperparameters.sgd_lr)

    if args.dataset is not None:
        if 'cbis' in args.dataset:
            dataset = cbis_ddsm_data_set
        if 'cbis_seg' in args.dataset:
            dataset = cbis_seg_data_set
        if 'bcs' in args.dataset:
            dataset = bcs_data_set

    if args.augmentation == 'True' or args.augmentation == 'true':
        hyperparameters.augmentation = True

    if args.tf_fit == 'True' or args.tf_fit == 'true':
        hyperparameters.tf_fit = True

    if args.preloaded_weights == 'True' or args.preloaded_weights == 'true':
        hyperparameters.preloaded_weights = True

    return hyperparameters, opt, dataset


def create_file_title(model_name, hyperparameters):
    meta_heuristic = hyperparameters.meta_heuristic if hyperparameters.meta_heuristic != None else 'none'
    order = hyperparameters.meta_heuristic_order if hyperparameters.meta_heuristic_order != None else 'na'
    return hyperparameters.experiment_id + '_' + model_name + '_' + meta_heuristic + '_' + order


