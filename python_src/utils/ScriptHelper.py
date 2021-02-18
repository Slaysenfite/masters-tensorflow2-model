from argparse import ArgumentParser

from tensorboard import program
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD

from configurations.TrainingConfig import output_dir
from metrics.MetricResults import MetricResult
from metrics.PerformanceMetrics import PerformanceMetrics


def generate_script_report(H, model, test_x, test_y, predictions, time_taken, data_set, hyperparameters, model_name,
                           metadata_string=''):
    metric_result = MetricResult(model, H, test_x, test_y, predictions, data_set)
    performance_metrics = PerformanceMetrics(time_taken=time_taken)
    with open(output_dir + model_name + '_' + data_set.name + '_' + metadata_string + '_result_report.txt',
              'w+') as text_file:
        text_file.write(
            hyperparameters.report_hyperparameters()
            + performance_metrics.report_metrics()
            + metric_result.report_result()
        )
    print(metric_result.report_result())


def read_cmd_line_args(hyperparameters):
    parser = ArgumentParser()
    parser.add_argument('--meta_heuristic', type=str)
    parser.add_argument('--meta_heuristic_order', type=str)
    parser.add_argument('--optimizer', type=str)
    args = parser.parse_args()
    if (args.meta_heuristic != None):
        hyperparameters.meta_heuristic = args.meta_heuristic

    if (args.meta_heuristic_order != None and args.meta_heuristic_order == 'first'):
        hyperparameters.meta_heuristic_order = 'first'
    elif (args.meta_heuristic_order != None and args.meta_heuristic_order == 'last'):
        hyperparameters.meta_heuristic_order = 'last'

    if (args.optimizer != None and args.optimizer == 'adam'):
        hyperparameters.learning_optimization = 'Adam'
        opt = Adam(learning_rate=hyperparameters.init_lr, decay=True)
    else:
        hyperparameters.learning_optimization = 'Stochatic Gradient Descent'
        opt = SGD(lr=hyperparameters.init_lr, momentum=0.9)

    return hyperparameters, opt


def create_file_title(model_name, hyperparameters):
    meta_heuristic = hyperparameters.meta_heuristic if hyperparameters.meta_heuristic != None else 'none'
    order = hyperparameters.meta_heuristic_order if hyperparameters.meta_heuristic_order != None else 'na'
    return model_name + '_' + meta_heuristic + '_' + order

def launch_tensorboard(log_dir):
    """
    Runs tensorboard with the given log_dir and wait for user input to kill the app.
    :param log_dir:
    :param clear_on_exit: If True Clears the log_dir on exit and kills the tensorboard app
    :return:
    """
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir])
    print("Launching Tensorboard ")
    url = tb.launch()
    print(url)

