import cv2
import numpy as np
from matplotlib import pyplot
from numpy import ma, asarray


def load_rgb_images(data, labels, dataset, image_dimensions=(128, 128, 3)):
    #get image paths and metadata
    image_paths = dataset.get_image_paths()
    metadata = dataset.get_image_metadata()

    i = 0
    # loop over the input images
    for image_path in image_paths:
        print_progress_bar(i + 1, len(image_paths), prefix=' Progress:', suffix='Complete')

        image = cv2.imread(image_path)
        image = cv2.resize(image, (image_dimensions[1], image_dimensions[0]))
        data.append(image)

        # extract the class label from the image path and update the
        # labels list
        raw_label = metadata[i][dataset.class_label_index]
        label = dataset.label_map[raw_label]

        labels.append(label)
        i += 1

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float32") / 255.0
    labels = np.array(labels)

    print("[INFO] Data shape: " + str(data.shape))
    print("[INFO] Label shape: " + str(labels.shape))

    return data, labels


def load_greyscale_images(data, labels, dataset, image_dimensions=(128, 128, 1)):
    # get image paths
    image_paths = dataset.get_image_paths()
    metadata = dataset.get_image_metadata()

    i = 0
    # loop over the input images
    for image_path in image_paths:
        print_progress_bar(i + 1, len(image_paths), prefix=' Progress:', suffix='Complete')

        image = cv2.imread(image_path)
        image = cv2.resize(image, (image_dimensions[1], image_dimensions[0]))
        image = cv2.cv2.cvtColor(image, cv2.COLOR_RGB2GRAY, dstCn=0)
        image = image[:, :, np.newaxis]
        data.append(image)

        # extract the class label from the image path and update the
        # labels list
        raw_label = metadata[i][dataset.class_label_index]
        label = dataset.label_map[raw_label]

        labels.append(label)
        i += 1

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    print("[INFO] Data shape: " + str(data.shape))
    print("[INFO] Label shape: " + str(labels.shape))

    return data, labels

def  load_seg_images(dataset, path_suffix='full', image_dimensions=(128, 128, 3)):
    # initialize the data and labels
    data = []
    labels = []

    # get image paths
    if (path_suffix=='cropped'):
        image_paths = dataset.get_cropped_image_paths()
        metadata = dataset.get_cropped_image_metadata()
    elif (path_suffix=='roi'):
        image_paths = dataset.get_roi_image_paths()
        metadata = dataset.get_roi_image_metadata()
    else:
        image_paths = dataset.get_image_paths()
        metadata = dataset.get_image_metadata()


    i = 0
    # loop over the input images
    for image_path in image_paths:
        print_progress_bar(i + 1, len(image_paths), prefix=' Progress:', suffix='Complete')

        image = cv2.imread(image_path)
        image = cv2.resize(image, (image_dimensions[1], image_dimensions[0]))
        data.append(image)

        # extract the class label from the image path and update the
        # labels list
        raw_label = metadata[i][dataset.class_label_index]
        label = dataset.label_map[raw_label]

        labels.append(label)
        i += 1

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    print("[INFO] Data shape: " + str(data.shape))
    print("[INFO] Label shape: " + str(labels.shape))

    return data, labels


def supplement_training_data(aug, train_x, train_y, multiclass=True):
    abnormal_data = []
    abnormal_labels = []

    if multiclass:
        append_three_class(abnormal_data, abnormal_labels, train_x, train_y)
    else:
        append_two_class(abnormal_data, abnormal_labels, train_x, train_y)
    append_three_class(abnormal_data, abnormal_labels, train_x, train_y)
    aug_output = aug.flow(asarray(abnormal_data), asarray(abnormal_labels), batch_size=len(abnormal_data),
                          shuffle=False)
    return ma.concatenate([train_x, aug_output.x]), ma.concatenate([train_y, aug_output.y])


def append_three_class(abnormal_data, abnormal_labels, train_x, train_y):
    for i, label in enumerate(train_y):
        if label == 0 or label == 1:
            abnormal_data.append(train_x[i])
            abnormal_labels.append(train_y[i])


def append_two_class(abnormal_data, abnormal_labels, train_x, train_y):
    for i, label in enumerate(train_y):
        if label == 0:
            abnormal_data.append(train_x[i])
            abnormal_labels.append(train_y[i])


def show_examples(title, train_x, test_x, train_y, test_y, items=9):
    print('Train: X=%s, y=%s' % (train_x.shape, train_y.shape))
    print('Test: X=%s, y=%s' % (test_x.shape, test_y.shape))
    # plot first few images
    for i in range(items):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # plot raw pixel data
        pyplot.imshow(train_x[i], cmap=pyplot.get_cmap('gray'))
    # show the figure
    pyplot.suptitle(title, fontsize=16)
    pyplot.show()
    pyplot.savefig(title+'.png')

# Print iterations progress
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
