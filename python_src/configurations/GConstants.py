import os


def create_required_directories():
    os.makedirs(output_dir, 0o777, True)
    os.makedirs(output_dir + 'figures/', 0o777, True)
    os.makedirs(output_dir + 'model/', 0o777, True)


IMAGE_DIMS = (256, 256, 3)

output_dir = 'output/'

FIGURE_OUTPUT = output_dir + 'figures/'
MODEL_OUTPUT = output_dir + 'model/'
