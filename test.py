import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer


class Detector(object):

    def __init__(self, net, weight_file):
        self.net = net
        self.weights_file = weight_file

        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)          # C = 20
        self.image_size = cfg.IMAGE_SIZE            # 448
        self.cell_size = cfg.CELL_SIZE              # S = 7
        self.boxes_per_cell = cfg.BOXES_PER_CELL    # B = 2
        self.threshold = cfg.THRESHOLD              # 0.2
        self.iou_threshold = cfg.IOU_THRESHOLD      # 0.5
        self.boundary1 = self.cell_size * self.cell_size * self.num_class   
            # S x S x C = 980
        self.boundary2 = self.boundary1 +\
            self.cell_size * self.cell_size * self.boxes_per_cell
            # S x S x C + S x S x B = 980 + 98 = 1078

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # restore weights file
        print('Restoring weights from: ' + self.weights_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)

    def detect(self, img):
        img_h, img_w, _ = img.shape     # img_h, img_w, _
        inputs = cv2.resize(img, (self.image_size, self.image_size))
            # (448, 448, 3)
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0       # normalization
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))
            # (1, 448, 448, 3)

        result = self.detect_from_cvmat(inputs)[0]
        
        for i in range(len(result)):
            result[i][1] *= (img_w / self.image_size)
            result[i][2] *= (img_h / self.image_size)
            result[i][3] *= (img_w / self.image_size)
            result[i][4] *= (img_h / self.image_size)

        return result

    def detect_from_cvmat(self, inputs):
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: inputs})
            # _, 1470 = S x S x (B * 5 + C)

        results = []
        for i in range(net_output.shape[0]):
            results.append(self.interpret_output(net_output[i]))

        return results

    def interpret_output(self, output):
    
        probs = np.zeros((self.cell_size, self.cell_size,
                          self.boxes_per_cell, self.num_class))
            # (S, S, B, C)
        
        # Conditional Class Probablity, Pr(Class_i|Object)    
        class_probs = np.reshape(
            output[0:self.boundary1],
            (self.cell_size, self.cell_size, self.num_class))
            # (S, S, C)
        
        # Confidence Score, Pr(Object)
        scales = np.reshape(
            output[self.boundary1:self.boundary2],
            (self.cell_size, self.cell_size, self.boxes_per_cell))
            # (S, S, B)
        
        # Bounding Box, (x, y, w, h) in the range [0, 1]    
        boxes = np.reshape(
            output[self.boundary2:],
            (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
            # (S, S, B, 4)
        
        # interpret network output (x, y, w, h) using offset
        offset = np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell)
            # (B * S, S)
        offset = np.reshape(
            offset,
            [self.boxes_per_cell, self.cell_size, self.cell_size])
            # (B, S, S)
        offset = np.transpose(offset, (1, 2, 0))
            # (S, S, B)
        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] /= self.cell_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])
        boxes *= self.image_size

        # Class Specific Confidence Score
        for i in range(self.boxes_per_cell):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i])

        # filtering via class specific confidence score
        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)

        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(
            probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
        
        # Since filter_mat_probs has boolean values, the unexpected results might be occurred. 
        # classes_num_filtered = np.argmax(
        #    filter_mat_probs, axis=3)[
        #    filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
        
        # non-maximal suppression
        # step-1: performing descending sort along class specific confidence score 
        argsort = np.argsort(probs_filtered)[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        # step-2: filtering via iou
        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append(
                [self.classes[classes_num_filtered[i]],
                 boxes_filtered[i][0],
                 boxes_filtered[i][1],
                 boxes_filtered[i][2],
                 boxes_filtered[i][3],
                 probs_filtered[i]])
            # (class, x, y, w, h, score)

        return result

    def iou(self, box1, box2):
        lr = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        tb = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        inter = 0 if lr < 0 or tb < 0 else lr * tb
        union = box1[2] * box1[3] + box2[2] * box2[3] - inter
        return inter / union

    def draw_result(self, img, result):
        for i in range(len(result)):
            x1 = int(result[i][1]) - int(result[i][3] / 2)
            y1 = int(result[i][2]) - int(result[i][4] / 2)
            x2 = int(result[i][1]) + int(result[i][3] / 2)
            y2 = int(result[i][2]) + int(result[i][4] / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y1 - 20),
                          (x2, y1), (125, 125, 125), -1)
            lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
            cv2.putText(
                img, result[i][0] + ' : %.3f' % result[i][5],
                (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, lineType)
            
    def image_detector(self, imname, wait=0):
        detect_timer = Timer()
        image = cv2.imread(imname)

        detect_timer.tic()
        result = self.detect(image)
        detect_timer.toc()
        print('Average detecting time: {:.3f}s'.format(
            detect_timer.average_time))

        self.draw_result(image, result)
        cv2.imshow('Image', image)
        cv2.waitKey(wait)

    def camera_detector(self, cap, wait=10):
        detect_timer = Timer()
        ret, _ = cap.read()

        while ret:
            ret, frame = cap.read()
            detect_timer.tic()
            result = self.detect(frame)
            detect_timer.toc()
            print('Average detecting time: {:.3f}s'.format(
                detect_timer.average_time))

            self.draw_result(frame, result)
            cv2.imshow('Camera', frame)
            cv2.waitKey(wait)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--weight_dir', default='weights', type=str)
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()

    if args.gpu is not None:
        cfg.GPU = args.gpu
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    yolo = YOLONet(False)
    weight_file = os.path.join(args.data_dir, args.weight_dir, args.weights)
    detector = Detector(yolo, weight_file)

    # detect from image file
    imname = 'person.jpg'
    detector.image_detector(imname)
    
    # detect from camera
    #cap = cv2.VideoCapture(-1)
    #detector.camera_detector(cap)


if __name__ == '__main__':
    main()
