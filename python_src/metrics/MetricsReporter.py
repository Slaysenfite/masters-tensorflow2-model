import itertools
import pickle
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve, auc

import configurations.TrainingConfig as constants

NETWORK_METRIC_PLOT = '_network_metric_plot.png'
ROC_PLOT = '_roc_plot.png'
CONFUSION_MATRIX_PLOT = '_confusion_matrix_plot.png'


class MetricReporter:
    def __init__(self, dataset_name, model_name):
        self.dataset_name = dataset_name
        self.model_name = model_name

    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Greens):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print('Normalized confusion matrix')
        else:
            print('Confusion matrix, without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(constants.FIGURE_OUTPUT + self.model_name + '_' + self.dataset_name + CONFUSION_MATRIX_PLOT)

    def plot_roc(self, class_names, test_y, predictions):
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(class_names)):
            fpr[i], tpr[i], _ = roc_curve(test_y[:, i], predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr['micro'], tpr['micro'], _ = roc_curve(test_y.ravel(), predictions.ravel())
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

        lw = 2

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_names))]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(class_names)):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= len(class_names)

        fpr['macro'] = all_fpr
        tpr['macro'] = mean_tpr
        roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr['micro'], tpr['micro'],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc['micro']),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr['macro'], tpr['macro'],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc['macro']),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(len(class_names)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class ' + class_names[i] + ' (area = {1:0.2f})'
                                                                    ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristics of detection task ')

        # Put a legend below current axis
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2)
        plt.legend(loc='lower right')
        plt.savefig(constants.FIGURE_OUTPUT + self.model_name + '_' + self.dataset_name + ROC_PLOT)

    def plot_network_metrics(self, H, model_name):
        # plot the training loss and accuracy
        N = np.arange(0, len(H.history['val_loss']))
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(N, H.history['loss'], label='train_loss')
        plt.plot(N, H.history['val_loss'], label='val_loss')
        plt.plot(N, H.history['accuracy'], label='train_acc')
        plt.plot(N, H.history['val_accuracy'], label='val_acc')
        plt.title('Training Loss and Accuracy (' + model_name + ')')
        plt.xlabel('Epoch #')
        plt.ylabel('Loss/Accuracy')
        plt.legend()
        plt.savefig(constants.FIGURE_OUTPUT + self.model_name + '_' + self.dataset_name + NETWORK_METRIC_PLOT)

    def save_model_to_file(self, model, lb):
        # save the model and label binarizer to disk
        model.save(constants.MODEL_OUTPUT + '/model' + '_' + self.dataset_name + '_' + self.model_name)
        f = open(constants.MODEL_OUTPUT + '/model' + '_' + self.model_name + '_' + self.dataset_name + '_label_bin',
                 'wb')
        f.write(pickle.dumps(lb))
        f.close()
