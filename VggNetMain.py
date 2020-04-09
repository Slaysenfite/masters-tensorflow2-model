import sys
import warnings

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from model.Hyperparameters import hyperparameters
from utils.ImageLoader import load_images
from model.DataSet import ddsm_data_set
from networks.VggNet16 import SmallVGGNet
from configurations.GlobalConstants import IMAGE_DIMS

# warnings.filterwarnings('ignore')



print('Python version: ' + sys.version + '\n')
print('[BEGIN] Start script...\n')
print('[INFO] Model hyperparameters...')
print(' Epochs: {}'.format(hyperparameters.epochs))
print(' Initial learning rate: {}'.format(hyperparameters.init_lr))
print(' Batch size: {}'.format(hyperparameters.batch_size))
print(' Image dimensions: {}\n'.format(IMAGE_DIMS))

# initialize the data and labels
data = []
labels = []


data, labels = load_images(data, labels, ddsm_data_set, IMAGE_DIMS)

# partition the data into training and testing splits using 70% of
# the data for training and the remaining 30% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.3, train_size=0.7, random_state=42)

# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct the image generator for data augmentation
print('[INFO] Augmenting data set')
aug = ImageDataGenerator()

model = SmallVGGNet.build(IMAGE_DIMS[0], IMAGE_DIMS[1], IMAGE_DIMS[2], classes=len(lb.classes_))

print('[INFO] Model summary...')
model.summary()

opt = SGD(lr=hyperparameters.init_lr, decay=hyperparameters.init_lr / hyperparameters.epochs)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# train the network
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=hyperparameters.batch_size),
                        validation_data=(testX, testY), steps_per_epoch=len(trainX) // hyperparameters.batch_size,
                        epochs=hyperparameters.epochs)

# evaluate the network
print('[INFO] evaluating network...')

predictions = model.predict(testX, batch_size=32)

print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=ddsm_data_set.class_names))
