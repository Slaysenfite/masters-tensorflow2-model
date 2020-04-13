IMAGE_DIMS = (8, 8, 3)

output_dir = 'output/figures/'

filename_prefix = output_dir + "ddsm" + '_' + "vggnet" + "_"

TERMINAL_OUTPUT_TXT = filename_prefix + 'terminal_output.txt'
NETWORK_METRIC_PLOT = filename_prefix + 'network_metric_plot.png'
ROC_PLOT = filename_prefix + 'roc_plot.png'
CONFUSION_MATRIX_PLOT = filename_prefix + 'confusion_matrix_plot.png'

MODEL_DUMP = 'bin/model-dump/' + 'confusion_matrix_plot.png'