from matplotlib import pyplot
from numpy import mean, std, arange
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.python.keras.metrics import Precision, Recall
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.utils.np_utils import to_categorical

from configurations.DataSet import mnist_data_set as data_set
from configurations.TrainingConfig import create_callbacks, create_required_directories, FIGURE_OUTPUT, output_dir
from configurations.TrainingConfig import mnist_hyperparameters as hyperparameters
from metrics.MetricsReporter import MetricReporter
from training_loops.CustomCallbacks import RunMetaHeuristicOnPlateau
from training_loops.CustomTrainingLoop import training_loop
from utils.ScriptHelper import generate_script_report, read_cmd_line_args

print('[BEGIN] Start script...\n')
hyperparameters, opt, data_set = read_cmd_line_args(hyperparameters, data_set)
print(hyperparameters.report_hyperparameters())

print('[INFO] Creating required directories...')
create_required_directories()

def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


train_x, train_y, test_x, test_y = load_dataset()


# # summarize loaded dataset
# print('Train: X=%s, y=%s' % (train_x.shape, train_y.shape))
# print('Test: X=%s, y=%s' % (test_x.shape, test_y.shape))
# # plot first few images
# for i in range(9):
#     # define subplot
#     pyplot.subplot(330 + 1 + i)
#     # plot raw pixel data
#     pyplot.imshow(train_x[i], cmap=pyplot.get_cmap('gray'))
# # show the figure
# pyplot.show()


# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


train_x, test_x = prep_pixels(train_x, test_x)

print("[INFO] Training data shape: " + str(train_x.shape))
print("[INFO] Training label shape: " + str(train_y.shape))

print('[INFO] Adding callbacks')
callbacks = create_callbacks()


def define_model(input=(28, 28, 1), classes=10):
    input = Input(shape=input)
    x = Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_uniform')(input)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_initializer='he_uniform')(x)
    output = Dense(classes, activation='softmax')(x)
    model = Model(inputs=input, outputs=output)

    opt = SGD(lr=hyperparameters.init_lr, momentum=0.9)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])
    return model, opt


# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    k = 1
    # enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
        print('K-Fold: {}'.format(k))
        # define model
        model, opt = define_model()
        # select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # fit model
        # Setup callbacks
        callbacks = create_callbacks()

        if hyperparameters.meta_heuristic != 'none':
            meta_callback = RunMetaHeuristicOnPlateau(
                X=train_x, y=train_y, meta_heuristic=hyperparameters.meta_heuristic, population_size=5, iterations=10,
                monitor='val_loss', factor=0.2, patience=0, verbose=1, mode='min',
                min_delta=0.1, cooldown=0)
            callbacks.append(meta_callback)
        if hyperparameters.tf_fit:
            print('Fit brah')
            history = model.fit(train_x, train_y, batch_size=hyperparameters.batch_size, validation_data=(test_x, test_y),
                          steps_per_epoch=len(train_x) // hyperparameters.batch_size, epochs=hyperparameters.epochs,
                          callbacks=callbacks)
        else:
            history = training_loop(model, opt, hyperparameters, train_x, train_y, test_x, test_y,
                                    meta_heuristic=hyperparameters.meta_heuristic,
                                    meta_heuristic_order=hyperparameters.meta_heuristic_order)
            # evaluate model
        acc = model.evaluate(testX, testY)
        print(str(model.metrics_names))
        print(str(acc))
        # stores scores
        scores.append(acc)
        histories.append(history)

        predictions = model.predict(testX, batch_size=hyperparameters.batch_size)
        generate_script_report(history, testY, predictions, data_set, hyperparameters, 'mnistnet', '-fold-' + str(k))
        reporter = MetricReporter("mnist", 'testnet-control', '-fold-' + str(k))

        cm1 = confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1))
        class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
        reporter.plot_confusion_matrix(cm1, classes=class_names, title='Confusion matrix, without normalization')

        reporter.plot_roc(class_names, testY, predictions)

        k = k + 1
    return scores, histories


# plot diagnostic learning curves
def summarize_diagnostics(histories):
    pyplot.style.use('ggplot')
    for i in range(len(histories)):
        N = arange(0, len(histories[i].history['val_loss']))
        pyplot.plot(N, histories[i].history['loss'], label='training loss ' + str(i+1))
        pyplot.plot(N, histories[i].history['val_loss'], label='validation loss ' + str(i+1))
        pyplot.plot(N, histories[i].history['accuracy'], label='training acc ' + str(i+1))
        pyplot.plot(N, histories[i].history['val_accuracy'], label='validation acc ' + str(i+1))
        pyplot.title('Training Loss and Accuracy')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Loss/Accuracy')
    pyplot.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    pyplot.tight_layout()
    pyplot.savefig(FIGURE_OUTPUT + 'network-diagnostics.png', bbox_inches='tight')
    pyplot.clf()


# summarize model performance
def summarize_performance(scores):
    # print summary
    pyplot.style.use('ggplot')
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores) * 100, std(scores) * 100, len(scores)))
    # box and whisker plots of results
    pyplot.boxplot(scores)
    pyplot.savefig(FIGURE_OUTPUT + 'box-and-whisker.png', bbox_inches='tight')
    pyplot.clf()


# run the test harness for evaluating a model
def process_scores(scores):
    ave_dict = {}
    ave_loss = []
    ave_acc = []
    ave_precision = []
    ave_recall = []
    for s in scores:
        ave_loss.append(s[0])
        ave_acc.append(s[1])
        ave_precision.append(s[2])
        ave_recall.append(s[3])
    ave_dict['val_loss'] = mean(ave_loss)
    ave_dict['val_accuracy'] = mean(ave_acc)
    ave_dict['val_precision'] = mean(ave_precision)
    ave_dict['val_recall'] = mean(ave_recall)

    with open(output_dir +'average_scores.txt', 'w+') as text_file:
        text_file.write(average_history_printer(ave_dict))

def average_history_printer(ave_dict):
    out = ''
    for i, metric in enumerate(ave_dict):
        if 'loss' in metric:
            out += '{} : {:.4f}\n'.format(metric, ave_dict[metric])
        else:
            out += '{} : {:.4f}%\n'.format(metric, ave_dict[metric]*100)
    return out

# evaluate model
scores, histories = evaluate_model(train_x, train_y)
# learning curves
summarize_diagnostics(histories)
# summarize estimated performance
summarize_performance(scores)
process_scores(scores)

print('[END] Finishing script...\n')
