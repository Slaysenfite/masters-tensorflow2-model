from tensorflow.python.keras.applications.inception_v3 import InceptionV3

from configurations.GConstants import IMAGE_DIMS
from networks.ResNet import resnet50
from networks.UNet import UNet
from networks.VggNet import Vgg19Net, SmallVggNet
from training_loops.OptimizerHelper import VggOneBlockFunctional

model = Vgg19Net.build(IMAGE_DIMS[0], IMAGE_DIMS[1], IMAGE_DIMS[2], classes=3)
print('VGG19Net')
model.summary()

model = SmallVggNet.build(IMAGE_DIMS[0], IMAGE_DIMS[1], IMAGE_DIMS[2], classes=3)
print('SmallVggNet')
model.summary()

model = InceptionV3(input_shape=IMAGE_DIMS, classes=3, weights=None)
print('InceptionV3')
model.summary()

model = UNet.build([IMAGE_DIMS[0], IMAGE_DIMS[1], 1], 3)
print('U-Net')
model.summary()

model = resnet50(IMAGE_DIMS, 3)
print('ResNet50')
model.summary()

model = VggOneBlockFunctional.build(IMAGE_DIMS[0], IMAGE_DIMS[1], IMAGE_DIMS[2], classes=3)
print('VggOneBlock')
model.summary()