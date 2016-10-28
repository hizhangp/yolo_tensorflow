import tensorflow as tf
import numpy as np
import datetime
import os
import argparse
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer
from utils.pascal_voc import pascal_voc


class Solver(object):

    def __init__(self, net, data):
        self.net = net
        self.data = data
        self.weights_file = cfg.WEIGHTS_FILE
        self.max_iter = cfg.MAX_ITER
        self.step_size = cfg.STEP_SIZE
        self.display_iter = cfg.DISPLAY_ITER
        self.save_iter = cfg.SAVE_ITER
        self.output_dir = os.path.join(cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # save configuration
        self.save_cfg()

        self.learning_rate = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.net.loss, global_step=self.global_step)
        self.summary_op = tf.merge_all_summaries()
        self.saver = tf.train.Saver(self.net.collection, max_to_keep=None)
        self.writer = tf.train.SummaryWriter(self.output_dir, flush_secs=10)
        self.ckpt_file = os.path.join(self.output_dir, 'save.ckpt')

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

        # load pre-trained model
        if self.weights_file is not None:
            print 'Restoring weights from: ' + self.weights_file
            self.saver.restore(self.sess, self.weights_file)

        # add graph to summary
        self.writer.add_graph(self.sess.graph)

    def train(self):

        # timer
        train_timer = Timer()
        load_timer = Timer()

        for step in xrange(1, self.max_iter + 1):
            learning_rate = cfg.LEARNING_RATE * (0.1 ** (step // self.step_size))
            # load data
            load_timer.tic()
            images, labels = self.data.get()
            load_timer.toc()
            feed_dict = {self.net.x: images, self.net.labels: labels, self.learning_rate: learning_rate}
            if step % self.display_iter == 0:
                # get summary
                train_timer.tic()
                summary_str, loss, _ = self.sess.run([self.summary_op, self.net.loss, self.optimizer],feed_dict=feed_dict)
                train_timer.toc()
                # write summary
                self.writer.add_summary(summary_str, global_step=self.global_step.eval(session=self.sess))
                print 'Step: {:5d}, Train Loss: {:6.2f}, Speed: {:.3f}s/iter, Time Remain: {}'.format(self.global_step.eval(session=self.sess), loss, train_timer.average_time, train_timer.remain(step, self.max_iter))
                # print 'Loading time: {:.3f}s/iter'.format(load_timer.average_time)
            else:
                # train
                train_timer.tic()
                loss, _ = self.sess.run([self.net.loss, self.optimizer],feed_dict=feed_dict)
                train_timer.toc()

            # save check point
            if step % self.save_iter == 0:
                print '[{}] Saving check point file to: {}'.format(datetime.datetime.now().strftime('%d/%m/%Y, %H:%M'), self.output_dir)
                self.saver.save(self.sess, self.ckpt_file, global_step=self.global_step)

    def save_cfg(self):

        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in cfg_dict:
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=None, type=str)
    parser.add_argument('--gpu', default=None, type=int)
    args = parser.parse_args()

    if args.weights is not None:
        cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, args.weights)
    if args.gpu is not None:
        cfg.GPU = str(args.gpu)

    # cfg.GPU = '0'
    # cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, 'YOLO_small.ckpt')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    yolo = YOLONet('train')
    pascal = pascal_voc('train')

    solver = Solver(yolo, pascal)

    solver.train()

if __name__ == '__main__':

    # python train.py --weights YOLO_small.ckpt --gpu 0
    main()
