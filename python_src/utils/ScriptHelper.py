from metrics.MetricResults import MetricResult
from configurations.GConstants import output_dir


def generate_script_report(H, test_y, predictions, data_set, hyperparameters):
    metric_result = MetricResult(H, test_y, predictions, data_set)
    with open('ddsm_vggnet' + output_dir + 'ddsm_vggnet_result_report.txt', 'w') as text_file:
        text_file.write(
            hyperparameters.report_hyperparameters() + metric_result.report_result()
        )
    print(metric_result.report_result())
