import sys
import warnings
import smtplib

from tf.keras.optimizers import SGD


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import src.config.GlobalConstants as gC
from src.model.Hyperparameters import hyperparameters
from src.utils.ImageLoader import load_images
from src.model.DataSet import ddsm_data_set


warnings.filterwarnings('ignore')

print('Python version: ' + sys.version + '\n')
print('[BEGIN] Start script...\n')
print('[INFO] Model hyperparameters...')
print(' Epochs: {}'.format(hyperparameters.epochs))
print(' Initial learning rate: {}'.format(hyperparameters.init_lr))
print(' Batch size: {}'.format(hyperparameters.batch_size))
print(' Image dimensions: {}\n'.format(gC.IMAGE_DIMS))


# initialize the data and labels
data = []
labels = []

data, labels = load_images(data, labels, config.path, config.label_map,
                                       config.arr_images, config.data_set, IMAGE_DIMS)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.3, random_state=42)

# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

