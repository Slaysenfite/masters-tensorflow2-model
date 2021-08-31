import gc

from tensorflow import GradientTape
from tensorflow.python.keras.metrics import Precision, Recall

from training_loops.GAOptimizer import GaEnv
from training_loops.OptimizerHelper import calc_solution_fitness
from training_loops.PsoOptimizer import PsoEnv
from training_loops.TrainingHelper import reset_metrics, \
    prepare_metrics, generate_tf_history


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


def apply_swarm_optimization(X, model, y, fitness_function, hyperparameters):
    pso = PsoEnv(num_solutions=30, iterations=30, model=model, X=X, y=y, fitness_function=fitness_function)
    pso.num_layers = hyperparameters.num_layers_for_optimization
    model = pso.get_optimized_model()
    return model


def apply_genetic_algorithm(X, model, y, fitness_function, hyperparameters):
    ga = GaEnv(num_solutions=30, iterations=30, model=model, X=X, y=y, fitness_function=fitness_function)
    ga.num_layers = hyperparameters.num_layers_for_optimization
    model = ga.get_optimized_model()
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
                  fitness_function=calc_solution_fitness,
                  task='binary_classification'
                  ):

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

        print('\rEpoch [%d/%d]  \n' % (epoch + 1, hyperparameters.epochs))

        # Run Algorithm
        model = run_meta_heuristic(meta_heuristic, model, train_x, train_y, fitness_function, hyperparameters)
        gc.collect()

        print(str(model.metrics_names))
        print(str(model.evaluate(test_x, test_y)))


        # Reset metrics
        reset_metrics(train_acc_metric, val_acc_metric)

    return generate_tf_history(model, hyperparameters, accuracy, loss, val_accuracy, val_loss, val_precision,
                               val_recall)


def run_meta_heuristic(meta_heuristic, model, train_x, train_y, fitness_function, hyperparameters):
    if (meta_heuristic != None and meta_heuristic == 'pso'):
        model = apply_swarm_optimization(train_x, model, train_y, fitness_function, hyperparameters)
    if (meta_heuristic != None and meta_heuristic == 'ga'):
        model = apply_genetic_algorithm(train_x, model, train_y, fitness_function, hyperparameters)
        return model
