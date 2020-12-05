from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.metrics import Precision, Recall, Accuracy
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.np_utils import to_categorical

from configurations.TrainingConfig import IMAGE_DIMS, hyperparameters, create_callbacks
from metrics.MetricsReporter import MetricReporter
from networks.ResNet import resnet50


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

model = resnet50(num_classes=10,
                 input_shape=(28, 28, 1))

opt = Adam(learning_rate=hyperparameters.init_lr, decay=True)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy',
                                                                       Precision(),
                                                                       Recall()])

print('[INFO] Adding callbacks')
callbacks = create_callbacks()

# train the network
H = model.fit(x=train_x, y=train_y, batch_size=hyperparameters.batch_size, validation_data=(test_x, test_y),
              steps_per_epoch=len(train_x) // hyperparameters.batch_size, epochs=hyperparameters.epochs,
              callbacks=callbacks)

# evaluate the network
print('[INFO] evaluating network...')

predictions = model.predict(test_x, batch_size=hyperparameters.batch_size)

print('[INFO] generating metrics...')

reporter = MetricReporter("mnist", 'testnet-control')
cm1 = confusion_matrix(test_y.argmax(axis=1), predictions.argmax(axis=1))
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
reporter.plot_confusion_matrix(cm1, classes=class_names,
                               title='Confusion matrix, without normalization')

reporter.plot_roc(class_names, test_y, predictions)

reporter.plot_network_metrics(H, 'testnet-control')

print('[END] Finishing script...\n')
