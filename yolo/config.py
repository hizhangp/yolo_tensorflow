import os

#
# path and dataset parameter
#

DATA_PATH = 'data'

# traning data
PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')

# label cache
CACHE_PATH = os.path.join(DATA_PATH, 'cache')

# output directory
OUTPUT_DIR = os.path.join(DATA_PATH, 'output')

# weight file directory
WEIGHTS_DIR = os.path.join(DATA_PATH, 'weights')

WEIGHTS_FILE = None

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
            'train', 'tvmonitor']

# flip image before training
FLIPPED = True


#
# model parameter
#

# fix image size (IMAGE_SIZE x IMAGE_SIZE)
IMAGE_SIZE = 448

CELL_SIZE = 7

BOXES_PER_CELL = 2

# leaky ReLU
ALPHA = 0.1

DISP_CONSOLE = False

# weight of loss
OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 0.5
CLASS_SCALE = 1.0
COORD_SCALE = 5.0


#
# solver parameter
#

GPU = ''

BATCH_SIZE = 32

LEARNING_RATE = 0.0001

# step size to change learning rate
STEP_SIZE = 4000

MAX_ITER = 20000

DISPLAY_ITER = 20

SAVE_ITER = 1000


#
# test parameter
#

# threshold for objectness
THRESHOLD = 0.2

IOU_THRESHOLD = 0.5
