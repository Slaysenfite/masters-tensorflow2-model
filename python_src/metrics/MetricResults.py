from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import operator


class MetricResult:
    # General metrics
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    report_dict = {}

    # Final network metric
    final_training_loss = 0.0
    final_validation_loss = 0.0
    final_training_accuracy = 0.0
    final_validation_accuracy = 0.0

    # Best network metric
    best_training_loss = 0.0
    best_validation_loss = 0.0
    best_training_accuracy = 0.0
    best_validation_accuracy = 0.0

    def __init__(self, H, test_y, predictions, data_set):
        self.H = H
        self.test_y = test_y
        self.predictions = predictions
        self.data_set = data_set
        self.get_general_metrics(test_y, predictions, data_set)
        self.get_best_network_metrics(H)
        self.get_final_network_metrics(H)

    def get_general_metrics(self, test_y, predictions, data_set):
        self.accuracy = accuracy_score(test_y.argmax(axis=1), predictions.argmax(axis=1)) * 100
        self.precision = precision_score(test_y.argmax(axis=1), predictions.argmax(axis=1), average='macro') * 100
        self.recall = recall_score(test_y.argmax(axis=1), predictions.argmax(axis=1), average='macro') * 100
        self.report_dict = classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1),
                                                 target_names=data_set.class_names)

    def get_best_network_metrics(self, H):
        idx, min_loss = min(enumerate(H.history['loss']), key=operator.itemgetter(1))

        self.best_training_loss = min_loss
        self.best_validation_loss = H.history['val_loss'][idx]
        self.best_training_accuracy = H.history['accuracy'][idx]
        self.best_validation_accuracy = H.history['val_accuracy'][idx]

    def get_final_network_metrics(self, H):
        self.final_training_loss = H.history['loss'][-1]
        self.final_validation_loss = H.history['val_loss'][-1]
        self.final_training_accuracy = H.history['accuracy'][-1]
        self.final_validation_accuracy = H.history['val_accuracy'][-1]

    def report_result(self):
        report = '*** Classification Report *** \n\n {} \n'.format(self.report_dict)

        report += '*** General Metrics *** \n'
        report += 'Accuracy Score: {:.2f}% \n'.format(self.accuracy)
        report += 'Precision Score: {:.2f}% \n'.format(self.precision)
        report += 'Recall Score: {:.2f}% \n\n'.format(self.recall)

        report += '*** Final Network Metrics *** \n'
        report += 'Final Training Loss: {:.2f}% \n'.format(self.final_training_loss)
        report += 'Final Validation Loss: {:.2f}% \n'.format(self.final_validation_loss)
        report += 'Final Training Accuracy: {:.2f}% \n'.format(self.final_training_accuracy)
        report += 'Final Validation Accuracy: {:.2f}% \n\n'.format(self.final_validation_accuracy)

        report += '*** Best Network Metrics *** \n'
        report += 'Best Training Loss: {:.2f}% \n'.format(self.best_training_loss)
        report += 'Best Validation Loss: {:.2f}% \n'.format(self.best_validation_loss)
        report += 'Best Training Accuracy: {:.2f}% \n'.format(self.best_training_accuracy)
        report += 'Best Validation Accuracy: {:.2f}% \n\n'.format(self.best_validation_accuracy)

        return report
