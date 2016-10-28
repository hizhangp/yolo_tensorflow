import numpy as np
import tensorflow as tf
import yolo.config as cfg


class YOLONet(object):

    def __init__(self, phase):
        self.weights_file = cfg.WEIGHTS_FILE

        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.output_size = (self.cell_size * self.cell_size) * (self.num_class + self.boxes_per_cell * 5)
        self.scale = 1.0 * self.image_size / self.cell_size
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell

        self.object_scale = cfg.OBJECT_SCALE
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE

        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size = cfg.BATCH_SIZE
        self.alpha = cfg.ALPHA
        self.disp_console = cfg.DISP_CONSOLE
        self.phase = phase
        self.collection = []
        self.offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell), (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        self.build_networks()


    def build_networks(self):
        if self.disp_console:
            print "Building YOLO_small graph..."
        self.x = tf.placeholder('float32', [None, 448, 448, 3])
        self.conv_1 = self.conv_layer(1, self.x, 64, 7, 2)
        self.pool_2 = self.pooling_layer(2, self.conv_1, 2, 2)
        self.conv_3 = self.conv_layer(3, self.pool_2, 192, 3, 1)
        self.pool_4 = self.pooling_layer(4, self.conv_3, 2, 2)
        self.conv_5 = self.conv_layer(5, self.pool_4, 128, 1, 1)
        self.conv_6 = self.conv_layer(6, self.conv_5, 256, 3, 1)
        self.conv_7 = self.conv_layer(7, self.conv_6, 256, 1, 1)
        self.conv_8 = self.conv_layer(8, self.conv_7, 512, 3, 1)
        self.pool_9 = self.pooling_layer(9, self.conv_8, 2, 2)
        self.conv_10 = self.conv_layer(10, self.pool_9, 256, 1, 1)
        self.conv_11 = self.conv_layer(11, self.conv_10, 512, 3, 1)
        self.conv_12 = self.conv_layer(12, self.conv_11, 256, 1, 1)
        self.conv_13 = self.conv_layer(13, self.conv_12, 512, 3, 1)
        self.conv_14 = self.conv_layer(14, self.conv_13, 256, 1, 1)
        self.conv_15 = self.conv_layer(15, self.conv_14, 512, 3, 1)
        self.conv_16 = self.conv_layer(16, self.conv_15, 256, 1, 1)
        self.conv_17 = self.conv_layer(17, self.conv_16, 512, 3, 1)
        self.conv_18 = self.conv_layer(18, self.conv_17, 512, 1, 1)
        self.conv_19 = self.conv_layer(19, self.conv_18, 1024, 3, 1)
        self.pool_20 = self.pooling_layer(20, self.conv_19, 2, 2)
        self.conv_21 = self.conv_layer(21, self.pool_20, 512, 1, 1)
        self.conv_22 = self.conv_layer(22, self.conv_21, 1024, 3, 1)
        self.conv_23 = self.conv_layer(23, self.conv_22, 512, 1, 1)
        self.conv_24 = self.conv_layer(24, self.conv_23, 1024, 3, 1)
        self.conv_25 = self.conv_layer(25, self.conv_24, 1024, 3, 1)
        self.conv_26 = self.conv_layer(26, self.conv_25, 1024, 3, 2)
        self.conv_27 = self.conv_layer(27, self.conv_26, 1024, 3, 1)
        self.conv_28 = self.conv_layer(28, self.conv_27, 1024, 3, 1)
        self.fc_29 = self.fc_layer(
            29, self.conv_28, 512, flat=True, linear=False)
        self.fc_30 = self.fc_layer(
            30, self.fc_29, 4096, flat=False, linear=False)
        if self.phase == 'train':
            self.dropout_31 = tf.nn.dropout(self.fc_30, keep_prob=0.5)
            self.fc_32 = self.fc_layer(
                32, self.dropout_31, self.output_size, flat=False, linear=True)
            self.labels = tf.placeholder('float32', [None, self.cell_size, self.cell_size, 5 + self.num_class])
            self.loss = self.loss_layer(33, self.fc_32, self.labels)
            tf.scalar_summary(self.phase + '/total_loss', self.loss)
        else:
            self.fc_32 = self.fc_layer(
                32, self.fc_30, self.output_size, flat=False, linear=True)

    def conv_layer(self, idx, inputs, filters, size, stride):
        channels = inputs.get_shape()[3]
        weight = tf.Variable(tf.truncated_normal(
            [size, size, int(channels), filters], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[filters]))
        self.collection.append(weight)
        self.collection.append(biases)

        pad_size = size // 2
        pad_mat = np.array([[0, 0], [pad_size, pad_size],
                            [pad_size, pad_size], [0, 0]])
        inputs_pad = tf.pad(inputs, pad_mat)

        conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1],
                            padding='VALID', name=str(idx) + '_conv')
        conv_biased = tf.add(conv, biases, name=str(idx) + '_conv_biased')

        if self.disp_console:
            print '    Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (
                idx, size, size, stride, filters, int(channels))
        return tf.maximum(self.alpha * conv_biased, conv_biased, name=str(idx) + '_leaky_relu')

    def pooling_layer(self, idx, inputs, size, stride):
        if self.disp_console:
            print '    Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (
                idx, size, size, stride)
        return tf.nn.max_pool(inputs, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME',
                              name=str(idx) + '_pool')

    def fc_layer(self, idx, inputs, hiddens, flat=False, linear=False):
        input_shape = inputs.get_shape().as_list()
        if flat:
            dim = input_shape[1] * input_shape[2] * input_shape[3]
            inputs_transposed = tf.transpose(inputs, (0, 3, 1, 2))
            inputs_processed = tf.reshape(inputs_transposed, [-1, dim])
        else:
            dim = input_shape[1]
            inputs_processed = inputs
        weight = tf.Variable(tf.truncated_normal([dim, hiddens], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[hiddens]))
        self.collection.append(weight)
        self.collection.append(biases)
        if self.disp_console:
            print '    Layer  %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' % (
                idx, hiddens, int(dim), int(flat), 1 - int(linear))
        if linear:
            return tf.add(tf.matmul(inputs_processed, weight), biases, name=str(idx) + '_fc')
        ip = tf.add(tf.matmul(inputs_processed, weight), biases)
        return tf.maximum(self.alpha * ip, ip, name=str(idx) + '_fc')


    def calc_iou(self, boxes1, boxes2):

        boxes1 = tf.pack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2, boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2,
                      boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2, boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2])
        boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

        boxes2 = tf.pack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2, boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2,
                        boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2, boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2])
        boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

        # calculate the left up point
        lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
        rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

        # intersection
        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

        # calculate the boxs1 square and boxs2 square
        square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
        square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

        # union
        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 1e-10, 1.0)

    def loss_layer(self, idx, predicts, labels):

        # predicted data
        predict_classes = tf.reshape(predicts[:, :self.boundary1], (self.batch_size, self.cell_size, self.cell_size, self.num_class))
        predict_scales = tf.reshape(predicts[:, self.boundary1:self.boundary2], (self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell))
        predict_boxes = tf.reshape(predicts[:, self.boundary2:], (self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4))

        # input labels
        response = tf.reshape(labels[:, :, :, 0], [self.batch_size, self.cell_size, self.cell_size, 1])
        boxes = tf.reshape(labels[:, :, :, 1:5], [self.batch_size, self.cell_size, self.cell_size, 1, 4])
        boxes = tf.tile(boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
        classes = labels[:, :, :, 5:]

        # transform predicted boxes to boxes format
        offset = tf.constant(self.offset, dtype=tf.float32)
        offset = tf.reshape(offset, [1, self.cell_size, self.cell_size, self.boxes_per_cell])
        offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
        predict_boxes_tran = tf.pack([(predict_boxes[:, :, :, :, 0] + offset) / self.cell_size,
                                      (predict_boxes[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) / self.cell_size,
                                      tf.square(predict_boxes[:, :, :, :, 2]),
                                      tf.square(predict_boxes[:, :, :, :, 3])])
        predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 4, 0])

        # calculate iou
        iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

        # object coordinate
        object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
        object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response

        # noobject coordinate
        noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

        # transform boxes to predicted boxes format
        boxes_tran = tf.pack([boxes[:, :, :, :, 0] * self.cell_size - offset,
                              boxes[:, :, :, :, 1] * self.cell_size - tf.transpose(offset, (0, 2, 1, 3)),
                              tf.sqrt(boxes[:, :, :, :, 2]),
                              tf.sqrt(boxes[:, :, :, :, 3])])
        boxes_tran = tf.transpose(boxes_tran, [1, 2, 3, 4, 0])

        # class_loss
        class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(response * (predict_classes - classes)),
                                                  reduction_indices=[1, 2, 3]) * self.class_scale, name='class_loss')

        # object_loss
        object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_mask * (predict_scales - iou_predict_truth)),
                                                   reduction_indices=[1, 2, 3]) * self.object_scale, name='object_loss')

        # noobject_loss
        noobject_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(noobject_mask * predict_scales), reduction_indices=[1, 2, 3]) * self.noobject_scale,
            name='noobject_loss')

        # coord_loss
        coord_mask = tf.reshape(object_mask, [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 1])
        boxes_delta = coord_mask * (predict_boxes - boxes_tran)
        coord_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(boxes_delta), reduction_indices=[1, 2, 3, 4]) * self.coord_scale, name='coord_loss')

        # iou loss
        iou_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(object_mask * iou_predict_truth), reduction_indices=[1, 2, 3]),
            name='iou_loss')

        # tensorboard summary
        tf.scalar_summary(self.phase + '/class_loss', class_loss)
        tf.scalar_summary(self.phase + '/object_loss', object_loss)
        tf.scalar_summary(self.phase + '/noobject_loss', noobject_loss)
        tf.scalar_summary(self.phase + '/coord_loss', coord_loss)
        tf.scalar_summary(self.phase + '/iou_loss', iou_loss)

        tf.histogram_summary(self.phase + '/boxes_delta_x', boxes_delta[:, :, :, :, 0])
        tf.histogram_summary(self.phase + '/boxes_delta_y', boxes_delta[:, :, :, :, 1])
        tf.histogram_summary(self.phase + '/boxes_delta_w', boxes_delta[:, :, :, :, 2])
        tf.histogram_summary(self.phase + '/boxes_delta_h', boxes_delta[:, :, :, :, 3])
        tf.histogram_summary(self.phase + '/iou', iou_predict_truth)

        return class_loss + object_loss + noobject_loss + coord_loss
