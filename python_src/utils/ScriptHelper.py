from argparse import ArgumentParser

from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD

from configurations.DataSet import cbis_ddsm_data_set, bcs_data_set, cbis_seg_data_set
from configurations.TrainingConfig import output_dir
from metrics.MetricResults import MetricResult
from metrics.MetricsReporter import MetricReporter
from metrics.PerformanceMetrics import PerformanceMetrics
from networks.NetworkHelper import generate_heatmap


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
    parser.add_argument('--id', type=str)
    parser.add_argument('--meta_heuristic', type=str)
    parser.add_argument('--meta_heuristic_layers', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--dropout_prob', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--augmentation', type=str)
    parser.add_argument('--preloaded_weights', type=str)
    parser.add_argument('--kernel_initializer', type=str)
    parser.add_argument('--preloaded_experiment', type=str)
    parser.add_argument('--tf_fit', type=str)
    parser.add_argument('--l2', type=float)
    args = parser.parse_args()

    if args.id is not None:
        hyperparameters.experiment_id = args.id

    if args.epochs is not None:
        hyperparameters.epochs = args.epochs

    if args.dropout_prob is not None:
        hyperparameters.dropout_prob = args.dropout_prob

    if args.meta_heuristic is not None:
        hyperparameters.meta_heuristic = args.meta_heuristic

    if args.meta_heuristic_layers is not None:
        hyperparameters.num_layers_for_optimization = args.meta_heuristic_layers

    if args.preloaded_experiment is not None:
        hyperparameters.weights_of_experiment_id = args.preloaded_experiment

    if args.kernel_initializer is not None:
        hyperparameters.kernel_initializer = args.kernel_initializer

    if args.l2 is not None:
        hyperparameters.l2 = args.l2

    if args.lr is not None:
        hyperparameters.lr = args.lr

    if args.optimizer is not None and args.optimizer == 'adam':
        hyperparameters.learning_optimization = 'Adam'
        opt = Adam(learning_rate=hyperparameters.lr)
    else:
        hyperparameters.learning_optimization = 'Stochastic Gradient Descent'
        opt = SGD(learning_rate=hyperparameters.lr)

    if args.dataset is not None:
        if 'cbis' in args.dataset:
            dataset = cbis_ddsm_data_set
        if 'cbis_seg' in args.dataset:
            dataset = cbis_seg_data_set
        if 'bcs' in args.dataset:
            dataset = bcs_data_set

    if args.augmentation == 'True' or args.augmentation == 'true':
        hyperparameters.augmentation = True

    if args.tf_fit == 'False' or args.tf_fit == 'false':
        hyperparameters.tf_fit = False

    if args.preloaded_weights == 'True' or args.preloaded_weights == 'true':
        hyperparameters.preloaded_weights = True

    return hyperparameters, opt, dataset


def create_file_title(model_name, hyperparameters):
    meta_heuristic = hyperparameters.meta_heuristic if hyperparameters.meta_heuristic != None else 'none'
    return hyperparameters.experiment_id + '_' + model_name + '_' + meta_heuristic

def evaluate_classification_model(model, model_name, hyperparameters, data_set, H, time_taken,test_x, test_y):
    print('[INFO] evaluating network...')
    acc = model.evaluate(test_x, test_y)
    print(str(model.metrics_names))
    print(str(acc))
    predictions = model.predict(test_x)
    print('[INFO] generating metrics...')
    file_title = create_file_title(model_name, hyperparameters)
    generate_script_report(H, model, test_x, test_y, predictions, time_taken, data_set, hyperparameters, file_title)
    reporter = MetricReporter(data_set.name, file_title)
    cm1 = confusion_matrix(test_y.argmax(axis=1), predictions.argmax(axis=1))
    reporter.plot_confusion_matrix(cm1, classes=data_set.class_names,
                                   title='Confusion matrix, without normalization')
    reporter.plot_roc(data_set.class_names, test_y, predictions)
    reporter.plot_network_metrics(H, file_title)

def evaluate_meta_model(model, model_name, hyperparameters, data_set, test_x, test_y):
    print('[INFO] evaluating network...')
    acc = model.evaluate(test_x, test_y)
    print(str(model.metrics_names))
    print(str(acc))
    predictions = model.predict(test_x)
    file_title = create_file_title(model_name, hyperparameters)
    reporter = MetricReporter(data_set.name, file_title)
    cm1 = confusion_matrix(test_y.argmax(axis=1), predictions.argmax(axis=1))
    reporter.plot_confusion_matrix(cm1, classes=data_set.class_names,
                                   title='Confusion matrix, without normalization')
    reporter.plot_roc(data_set.class_names, test_y, predictions)