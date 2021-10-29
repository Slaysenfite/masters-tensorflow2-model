import gc
import sys
import time
from datetime import timedelta

import tensorflow as tf
from tensorflow.python.keras.applications.xception import Xception

from configurations.DataSet import cbis_ddsm_data_set as data_set
from configurations.TrainingConfig import IMAGE_DIMS, create_required_directories, hyperparameters, create_callbacks, \
    MODEL_OUTPUT
from networks.NetworkHelper import create_classification_layers, compile_with_regularization, generate_heatmap
from training_loops.CustomCallbacks import RunMetaHeuristicOnPlateau
from training_loops.CustomTrainingLoop import training_loop
from utils.ScriptHelper import read_cmd_line_args, evaluate_classification_model, evaluate_meta_model

print('Python version: {}'.format(sys.version))
print('Tensorflow version: {}\n'.format(tf.__version__))
print('[BEGIN] Start script...\n')
hyperparameters, opt, data_set = read_cmd_line_args(hyperparameters, data_set)
print(' Dataset: {}\n'.format(data_set.name))
print(' Image dimensions: {}\n'.format(IMAGE_DIMS))
print(hyperparameters.report_hyperparameters())

print('[INFO] Creating required directories...')
create_required_directories()
gc.enable()

print('[INFO] Loading images...')
train_x, test_x, train_y, test_y = data_set.split_data_set(IMAGE_DIMS,
                                                           subset=None,
                                                           segment=hyperparameters.dataset_segment)
loss, train_y, test_y = data_set.get_dataset_labels(train_y, test_y)

if hyperparameters.preloaded_weights:
    print('[INFO] Loading imagenet weights')
    weights = 'imagenet'
else:
    weights = None

model = Xception(
    include_top=False,
    weights=weights,
    input_shape=IMAGE_DIMS,
    classes=len(data_set.class_names))

model = create_classification_layers(base_model=model,
                                     classes=len(data_set.class_names),
                                     dropout_prob=hyperparameters.dropout_prob)

if hyperparameters.weights_of_experiment_id is not None:
    path_to_weights = '{}{}.h5'.format(MODEL_OUTPUT, hyperparameters.weights_of_experiment_id)
    print('[INFO] Loading weights from {}'.format(path_to_weights))
    model.load_weights(path_to_weights)

# compile model
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

# train the network
start_time = time.time()

print('META-HEURISTIC')

H = training_loop(model, hyperparameters, train_x, train_y, test_x, test_y,
                  meta_heuristic=hyperparameters.meta_heuristic, num_solutions=20, iterations=5)
time_taken = timedelta(seconds=(time.time() - start_time))

print('EVALUATION')

generate_heatmap(model, test_x, 10, 0, hyperparameters, '_meta_pass')
generate_heatmap(model, test_x, 10, 1, hyperparameters, '_meta_pass')

# evaluate the network
evaluate_meta_model(model, 'XceptionMeta', hyperparameters, data_set, test_x, test_y)

print('[END] Finishing script...\n')


