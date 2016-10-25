import tensorflow as tf
import numpy as np
import datetime
import os
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

        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.net.loss)
        self.summary_op = tf.merge_all_summaries()
        self.saver = tf.train.Saver(self.net.collection)
        self.writer = tf.train.SummaryWriter(self.output_dir, flush_secs=10)
        self.ckpt_file = os.path.join(self.output_dir, 'save.ckpt')

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

        if self.weights_file is not None:
            print 'Restoring weights from: ' + self.weights_file
            self.saver.restore(self.sess, self.weights_file)

        self.writer.add_graph(self.sess.graph)

    def train(self):

        train_timer = Timer()
        load_timer = Timer()

        for step in xrange(1, self.max_iter + 1):
            learning_rate = cfg.LEARNING_RATE * (0.1 ** (step // self.step_size))
            load_timer.tic()
            images, labels = self.data.get()
            load_timer.toc()
            feed_dict = {self.net.x: images, self.net.labels: labels, self.learning_rate: learning_rate}
            if step % self.display_iter == 0:
                train_timer.tic()
                summary_str, loss, _ = self.sess.run([self.summary_op, self.net.loss, self.optimizer],feed_dict=feed_dict)
                train_timer.toc()
                self.writer.add_summary(summary_str, global_step=step)
                print 'Step: {:5d}, Train Loss: {:6.2f}, Speed: {:.3f}s/iter, Time Remain: {}'.format(step, loss, train_timer.average_time, train_timer.remain(step, self.max_iter))
                # print 'Loading time: {:.3f}s/iter'.format(load_timer.average_time)
            else:
                train_timer.tic()
                loss, _ = self.sess.run([self.net.loss, self.optimizer],feed_dict=feed_dict)
                train_timer.toc()

            if step % self.save_iter == 0:
                print '[{}] Saving check point file to: {}'.format(datetime.datetime.now().strftime('%d/%m/%Y, %H:%M'), self.output_dir)
                self.saver.save(self.sess, self.ckpt_file, global_step=step)

def main():
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    yolo = YOLONet('train')
    pascal = pascal_voc('train', True)

    solver = Solver(yolo, pascal)

    solver.train()

if __name__ == '__main__':

    main()
