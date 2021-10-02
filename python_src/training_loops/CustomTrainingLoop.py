import gc

from tensorflow.python.keras.metrics import Precision, Recall

from training_loops.GAOptimizer import GaEnv
from training_loops.OptimizerHelper import calc_solution_fitness
from training_loops.PsoOptimizer import PsoEnv
from training_loops.TrainingHelper import reset_metrics, \
    prepare_metrics, generate_tf_history, print_metrics, append_epoch_metrics, batch_data_set


def train_on_batch(model, X, y, accuracy_metric, loss_metric):
    ŷ = model(X, training=True)
    # Calculate loss after pso weight updating
    precision_metric = Precision()
    recall_metric = Recall()
    accuracy = accuracy_metric(y, ŷ)
    loss = loss_metric(y, ŷ)
    precision = precision_metric(y, ŷ)
    recall = recall_metric(y, ŷ)
    # Update training metric.
    return accuracy.numpy(), loss.numpy(), precision.numpy(), recall.numpy()


def apply_swarm_optimization(X, model, y, fitness_function, hyperparameters):
    pso = PsoEnv(num_solutions=30, iterations=5, model=model, X=X, y=y, fitness_function=fitness_function)
    pso.num_layers = hyperparameters.num_layers_for_optimization
    model = pso.get_optimized_model()
    return model


def apply_genetic_algorithm(X, model, y, fitness_function, hyperparameters):
    ga = GaEnv(num_solutions=30, iterations=5, model=model, X=X, y=y, fitness_function=fitness_function)
    ga.num_layers = hyperparameters.num_layers_for_optimization
    model = ga.get_optimized_model()
    return model


# The validate_on_batch function
# Find out how the model works
# @tf.function
def validate_on_batch(model, X, y, accuracy_metric, loss_metric):
    ŷ = model(X, training=False)
    precision_metric = Precision()
    recall_metric = Recall()
    accuracy = accuracy_metric(y, ŷ)
    loss = loss_metric(y, ŷ)
    precision = precision_metric(y, ŷ)
    recall = recall_metric(y, ŷ)
    return accuracy.numpy(), loss.numpy(), precision.numpy(), recall.numpy()


# The Custom Loop For The SGD-PSO based optimizer
def training_loop(model,
                  hyperparameters,
                  train_x,
                  train_y,
                  test_x,
                  test_y,
                  meta_heuristic=None,
                  fitness_function=calc_solution_fitness,
                  task='binary_classification',
                  meta_epochs=3
                  ):
    loss = []
    accuracy = []
    val_loss = []
    val_accuracy = []
    val_precision = []
    val_recall = []

    # Separate into batches
    test_data, train_data = batch_data_set(hyperparameters, test_x, test_y, train_x, train_y)

    # Enumerating the Dataset
    for epoch in range(0, meta_epochs):
        print('\rEpoch [%d/%d] \n' % (epoch + 1, meta_epochs), end='')

        # Prepare the metrics.
        train_acc_metric, train_loss_metric, val_acc_metric, val_loss_metric = prepare_metrics(task)

        for batch, (X, y) in enumerate(train_data):
            print('\r Batch: %d%s \n' % (batch, '.' * (batch % 10)),
                  end='')

            # Run Algorithm
            model = run_meta_heuristic(meta_heuristic, model, X, y, fitness_function, hyperparameters)
            gc.collect()
            train_acc_score, train_loss_score, train_precision_score, train_recall_score = train_on_batch(model,
                                                                                                          X, y,
                                                                                                          train_acc_metric,
                                                                                                          train_loss_metric)
            print('train_loss: {} train_acc: {} train_pre:{} train_rec: {}'.format(float(train_loss_score),
                                                                                   float(train_acc_score),
                                                                                   float(train_precision_score),
                                                                                   float(train_recall_score)))

        print('\rEpoch [%d/%d]  \n' % (epoch + 1, meta_epochs))

        # Run a validation loop at the end of each epoch.
        val_acc_score, val_loss_score, val_precision_score, val_recall_score = validate_on_batch(model, test_x, test_y,
                                                                                                 val_acc_metric,
                                                                                                 val_loss_metric)

        # Display metrics at the end of each epoch.
        print_metrics(val_acc_score, val_loss_score, val_precision_score, val_recall_score)

        # Append metrics
        append_epoch_metrics(val_acc_score, val_accuracy, val_loss, val_loss_score, val_precision, val_precision_score,
                             val_recall, val_recall_score)

        # Reset metrics
        reset_metrics(train_acc_metric, val_acc_metric)

    return generate_tf_history(model, hyperparameters, accuracy, loss, val_accuracy, val_loss, val_precision,
                               val_recall)


def run_meta_heuristic(meta_heuristic, model, train_x, train_y, fitness_function, hyperparameters):
    if meta_heuristic is 'pso':
        model = apply_swarm_optimization(train_x, model, train_y, fitness_function, hyperparameters)
    elif meta_heuristic is 'ga':
        model = apply_genetic_algorithm(train_x, model, train_y, fitness_function, hyperparameters)
    else:
        model = apply_genetic_algorithm(train_x, model, train_y, fitness_function, hyperparameters)
    return model
