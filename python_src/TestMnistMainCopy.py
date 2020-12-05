from matplotlib import pyplot
from numpy import mean, std
from sklearn.model_selection import KFold
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.utils.np_utils import to_categorical

from configurations.TrainingConfig import  create_callbacks
from configurations.TrainingConfig import mnist_hyperparameters as hyperparameters
from training_loops.CustomTrainingLoop import training_loop


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


def define_model():
    input = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform')(input)
    x = MaxPooling2D((2, 2))(input)
    x = Flatten()(x)
    x = Dense(100, activation='relu', kernel_initializer='he_uniform')(x)
    output = Dense(10, activation='softmax')(x)
    model = Model(inputs=input, outputs=output)

    # model = resnet50(num_classes=10,
    #                  input_shape=(28, 28, 1))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model, opt


# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
        # define model
        model, opt = define_model()
        # select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # fit model
        history = training_loop(model, opt, hyperparameters, train_x, train_y, test_x, test_y, pso_layer=(Conv2D, Dense),
                  gd_layer=(Conv2D, Dense))
        # evaluate model
        acc = model.evaluate(testX, testY)
        print(str(model.metrics_names))
        print(str(acc))
        # stores scores
        scores.append(acc)
        histories.append(history)
    return scores, histories


# plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(2, 1, 1)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    pyplot.show()


# summarize model performance
def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores) * 100, std(scores) * 100, len(scores)))
    # box and whisker plots of results
    pyplot.boxplot(scores)
    pyplot.show()


# run the test harness for evaluating a model
def run_test_harness():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # evaluate model
    scores, histories = evaluate_model(trainX, trainY)
    # learning curves
    summarize_diagnostics(histories)
    summarize_performance(scores)


# evaluate model
scores, histories = evaluate_model(train_x, train_y)
# learning curves
summarize_diagnostics(histories)
# summarize estimated performance
summarize_performance(scores)

print('[END] Finishing script...\n')
