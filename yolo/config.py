import os

### path and dataset parameter

DATA_PATH = 'data'

PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')

WEIGHTS_DIR = os.path.join(PASCAL_PATH, 'weights')

WEIGHTS_FILE = None
# WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')

OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')

CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')

CLASSES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
    ]

FLIPPED = True


### model parameter

IMAGE_SIZE = 448
CELL_SIZE = 7
BOXES_PER_CELL = 2

OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0

### in the paper, 
# OBJECT_SCALE = 1.0
# NOOBJECT_SCALE = 0.5
# CLASS_SCALE = 1.0
# COORD_SCALE = 5.0


### train/solver parameter

GPU = ''
#LEARNING_RATE = 0.0001
#DECAY_STEPS = 30000
#DECAY_RATE = 0.1
#STAIRCASE = True
#BATCH_SIZE = 45
#MAX_ITER = 15000
#SUMMARY_ITER = 10
#SAVE_ITER = 1000

LEARNING_RATE = 0.0001
DECAY_STEPS = 500
DECAY_RATE = 0.1
STAIRCASE = True
BATCH_SIZE = 64
MAX_ITER = 1000
SUMMARY_ITER = 10
SAVE_ITER = 500


### test parameter

THRESHOLD = 0.2

IOU_THRESHOLD = 0.5
