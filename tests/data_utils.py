import os

DATA_DIRECTORY = 'data'


def get_data_path(filename):
    test_directory = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(test_directory, DATA_DIRECTORY, filename)
