from tensorflow import GradientTape
from tensorflow.python.keras.metrics import Precision, Recall

from training_loops.GAOptimizer import GaEnv
from training_loops.OptimizerHelper import calc_solution_fitness
from training_loops.PsoOptimizer import PsoEnv
from training_loops.TrainingHelper import print_metrics, append_epoch_metrics, reset_metrics, \
    batch_data_set, prepare_metrics, generate_tf_history


def train_on_batch(model, optimizer, X, y, accuracy_metric, loss_metric):
    apply_gradient_descent(X, model, optimizer, y)
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


def apply_swarm_optimization(X, model, y, fitness_function):
    pso = PsoEnv(swarm_size=20, iterations=10, model=model, X=X, y=y, fitness_function=fitness_function)
    model = pso.get_pso_model()
    return model


def apply_genetic_algorithm(X, model, y, fitness_function):
    ga = GaEnv(population_size=20, iterations=10, model=model, X=X, y=y, fitness_function=fitness_function)
    model = ga.get_ga_model()
    return model


def apply_gradient_descent(X, model, optimizer, y):
    with GradientTape() as tape:
        ŷ = model(X, training=True)
        loss_value = model.compiled_loss(y, ŷ)
    gd_weights = model.trainable_variables
    grads = tape.gradient(loss_value, gd_weights)
    optimizer.apply_gradients(zip(grads, gd_weights))
    model.compiled_metrics.update_state(y, ŷ)
    model._trainable_variables = gd_weights

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
                  optimizer,
                  hyperparameters,
                  train_x,
                  train_y,
                  test_x,
                  test_y,
                  meta_heuristic=None,
                  meta_heuristic_order=None,
                  fitness_function=calc_solution_fitness,
                  task='binary_classification'
                  ):

    # Validate input params

    perform_input_validation(meta_heuristic, meta_heuristic_order)

    # Separate into batches
    test_data, train_data = batch_data_set(hyperparameters, test_x, test_y, train_x, train_y)

    loss = []
    accuracy = []
    val_loss = []
    val_accuracy = []
    val_precision = []
    val_recall = []

    # Enumerating the Dataset
    for epoch in range(0, hyperparameters.epochs):
        print('\rEpoch [%d/%d] \n' % (epoch + 1, hyperparameters.epochs), end='')

        # Prepare the metrics.
        train_acc_metric, train_loss_metric, val_acc_metric, val_loss_metric = prepare_metrics(task)

        run_meta_heuristic(meta_heuristic, 'first', meta_heuristic_order, model, train_x, train_y, fitness_function)

        print('\rEpoch [%d/%d]  \n' % (epoch + 1, hyperparameters.epochs))

        for batch, (X, y) in enumerate(train_data):
            # print('\rEpoch [%d/%d] Batch: %d%s \n' % (epoch + 1, hyperparameters.epochs, batch, '.' * (batch % 10)),
            #       end='')

            train_acc_score, train_loss_score, train_precision_score, train_recall_score = train_on_batch(model,
                                                                                                          optimizer,
                                                                                                          X, y,
                                                                                                          train_acc_metric,
                                                                                                          train_loss_metric)

        run_meta_heuristic(meta_heuristic, 'last', meta_heuristic_order, model, train_x, train_y, fitness_function)

        # Run a validation loop at the end of each epoch.
        for (X, y) in test_data:
            val_acc_score, val_loss_score, val_precision_score, val_recall_score = validate_on_batch(model, X, y,
                                                                                                     val_acc_metric,
                                                                                                     val_loss_metric)

        # Display metrics at the end of each epoch.
        print_metrics(train_acc_score, train_loss_score, train_precision_score, train_recall_score, val_acc_score,
                      val_loss_score, val_precision_score, val_recall_score)

        # Append metrics
        append_epoch_metrics(accuracy, loss, train_acc_score, train_loss_score, val_acc_score, val_accuracy, val_loss,
                             val_loss_score, val_precision, val_precision_score, val_recall, val_recall_score)

        # Reset metrics
        reset_metrics(train_acc_metric, val_acc_metric)

    return generate_tf_history(model, hyperparameters, accuracy, loss, val_accuracy, val_loss, val_precision,
                               val_recall)


def run_meta_heuristic(meta_heuristic, prefered_order, meta_heuristic_order, model, train_x, train_y, fitness_function):
    if (meta_heuristic != None and meta_heuristic == 'pso' and meta_heuristic_order == prefered_order):
        apply_swarm_optimization(train_x, model, train_y, fitness_function)
    if (meta_heuristic != None and meta_heuristic == 'ga' and meta_heuristic_order == prefered_order):
        apply_genetic_algorithm(train_x, model, train_y, fitness_function)


def perform_input_validation(meta_heuristic, meta_heuristic_order):
    if (meta_heuristic == None and meta_heuristic_order != None) or (
            meta_heuristic != None and meta_heuristic_order == None):
        raise Exception('Must specify meta_heuristic with order of execution')

