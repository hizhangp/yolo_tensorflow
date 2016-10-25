import os

DATA_PATH = 'data'

PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')

CACHE_PATH = os.path.join(DATA_PATH, 'cache')

OUTPUT_DIR = os.path.join(DATA_PATH, 'output')

WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')
# WEIGHTS_FILE = None

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
            'train', 'tvmonitor']

GPU = '0'

IMAGE_SIZE = 448

CELL_SIZE = 7

BOXES_PER_CELL = 2

OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 0.5
CLASS_SCALE = 1.0
COORD_SCALE = 5.0

PHASE = 'train'

LEARNING_RATE = 0.00001

STEP_SIZE = 4000

BATCH_SIZE = 32

MAX_ITER = 20000

DISPLAY_ITER = 20

SAVE_ITER = 1000

ALPHA = 0.1

DISP_CONSOLE = False

FLIPPED = True

THRESHOLD = 0.2

IOU_THRESHOLD = 0.5
