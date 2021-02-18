class PerformanceMetrics(object):
    time_taken = 0.0

    def __init__(self, time_taken=0, ):
        self.time_taken = time_taken

    def report_metrics(self):
        report = '*** PERFORMANCE METRICS ***\n'

        report += '\rTime taken: ' + str(self.time_taken)

        return report
