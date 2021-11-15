import pandas as pd

from configurations.DataSet import cbis_ddsm_data_set, binary_ddsm_data_set


class DataBalance:
    dataset_name = ''
    num_positive = 0
    num_negative = 0

    def __init__(self, num_positive, num_negative, dataset_name):
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.dataset_name = dataset_name

    def reporter(self):
        total = self.num_negative + self.num_positive

        print('{}'.format(self.dataset_name))
        print('positive: {} negative: {}'.format(self.num_positive, self.num_negative))
        print('positive %: {} negative % {}'.format(self.num_positive / total, self.num_negative / total))


def count_get_data_balance(dataset_name, metadata_arr):
    num_positive = 0
    num_negative = 0
    for i in metadata_arr:
        if i[1] == 'P':
            num_positive += 1
        if i[1] == 'N':
            num_negative += 1

    return DataBalance(num_positive, num_negative, dataset_name)


ddsm = count_get_data_balance(binary_ddsm_data_set.name, binary_ddsm_data_set.get_image_metadata())
cbis = count_get_data_balance(cbis_ddsm_data_set.name, cbis_ddsm_data_set.get_image_metadata())

ddsm.reporter()
cbis.reporter()
