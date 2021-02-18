import operator

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report


class MetricResult:
    # General metrics
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    report_dict = {}
    internal_metrics = None

    # Final network metric
    final_training_loss = 0.0
    final_validation_loss = 0.0
    final_training_accuracy = 0.0
    final_validation_accuracy = 0.0

    # Best network metric
    best_epoch = 0
    best_training_loss = 0.0
    best_validation_loss = 0.0
    best_training_accuracy = 0.0
    best_validation_accuracy = 0.0

    def __init__(self, model, H, test_x, test_y, predictions, data_set):
        self.H = H
        self.test_x = test_x
        self.test_y = test_y
        self.predictions = predictions
        self.data_set = data_set
        self.model = model
        self.get_general_metrics(test_y, predictions, data_set)
        self.get_best_network_metrics(H)
        self.get_final_network_metrics(H)
        self.get_model_internal_metrics(model, test_x, test_y)

    def get_general_metrics(self, test_y, predictions, data_set):
        self.accuracy = accuracy_score(test_y.argmax(axis=1), predictions.argmax(axis=1)) * 100
        self.precision = precision_score(test_y.argmax(axis=1), predictions.argmax(axis=1), average='macro') * 100
        self.recall = recall_score(test_y.argmax(axis=1), predictions.argmax(axis=1), average='macro') * 100
        self.report_dict = classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1),
                                                 target_names=data_set.class_names)

    def get_best_network_metrics(self, H):
        idx, min_loss = min(enumerate(H.history['val_loss']), key=operator.itemgetter(1))

        self.best_epoch = idx + 1
        self.best_training_loss = H.history['loss'][idx]
        self.best_validation_loss = H.history['val_loss'][idx]
        self.best_training_accuracy = H.history['accuracy'][idx]
        self.best_validation_accuracy = H.history['val_accuracy'][idx]

    def get_final_network_metrics(self, H):
        self.final_training_loss = H.history['loss'][-1]
        self.final_validation_loss = H.history['val_loss'][-1]
        self.final_training_accuracy = H.history['accuracy'][-1]
        self.final_validation_accuracy = H.history['val_accuracy'][-1]

    def get_model_internal_metrics(self, model, test_x, test_y):
        self.internal_metrics = model.evaluate(test_x, test_y)

    def report_result(self):
        report = '*** SCI-KIT METRICS ***\n\n'
        report += '*** Classification Report *** \n {} \n'.format(self.report_dict)

        report += '*** General Metrics *** \n'
        report += ' Accuracy Score: {:.2f}% \n'.format(self.accuracy)
        report += ' Precision Score: {:.2f}% \n'.format(self.precision)
        report += ' Recall Score: {:.2f}% \n\n'.format(self.recall)

        report += '*** Final Network Metrics *** \n'
        report += ' Final Training Loss: {:.4f} \n'.format(self.final_training_loss)
        report += ' Final Validation Loss: {:.4f} \n'.format(self.final_validation_loss)
        report += ' Final Training Accuracy: {:.2f}% \n'.format(self.final_training_accuracy)
        report += ' Final Validation Accuracy: {:.2f}% \n\n'.format(self.final_validation_accuracy)

        report += '*** Best Network Metrics *** \n'
        report += ' Best Epoch: {} \n'.format(self.best_epoch)
        report += ' Best Training Loss: {:.4f} \n'.format(self.best_training_loss)
        report += ' Best Validation Loss: {:.4f} \n'.format(self.best_validation_loss)
        report += ' Best Training Accuracy: {:.2f}% \n'.format(self.best_training_accuracy)
        report += ' Best Validation Accuracy: {:.2f}% \n\n'.format(self.best_validation_accuracy)

        report += '*** INTERNAL METRICS ***\n'
        report += str(self.model.metrics_names) + '\n'
        report += str(self.internal_metrics)

        report += '*** Predictions *** \n'
        report += ' Predictions: {} \n'.format(self.predictions)
        report += ' Ground Truth Labels: {} \n'.format(self.test_y)

        return report
