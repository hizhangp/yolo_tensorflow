## YOLO_tensorflow

Tensorflow implementation of [YOLO](https://arxiv.org/pdf/1506.02640.pdf), including training and testing phase.

### Installation

1. Clone YOLO_tensorflow repository
	```Shell
	git clone https://github.com/hizhangp/YOLO_tensorflow.git
  cd YOLO_tensorflow
	```

2. Download Pascal VOC dataset, we call the directory `PASCAL_VOC`
	```Shell
	ln -s $PASCAL_VOC data/pascal_voc
	```

3. Download [YOLO_small](https://drive.google.com/file/d/0B2JbaJSrWLpza08yS2FSUnV2dlE/view?usp=sharing)
weight file and put it in `data/weight`

4. Change setting in `yolo/config.py`

5. Training
	```Shell
	python train.py
	```

6. Training
	```Shell
	python test.py
	```

### Requirements
1. Tensorflow

2. OpenCV
