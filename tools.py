import time
import sys
class ProcessBar():
    def reset(self, length):
        self._length = length
        self._start = time.time()
    def show(self, i, msg=""):
        percents = (i + 1) / self._length
        equal_length = int(50 * percents) * "="
        empty_length = (49 - int(50 * percents)) * " "
        elapsed_time = time.time() - self._start
        eta_time = elapsed_time / percents * (1 - percents)
        line_str = "[{}>{}] {}/{} {:.1f}% ETA:{:.2f}s {:.2f}s {}"\
            .format(equal_length, empty_length, i, self._length,
                    100 * percents, eta_time, elapsed_time, msg)
        sys.stdout.write("\r" + line_str)
    def summary(self, i, msg=""):
        line_str = "[{}] {} {} {:.2f}s {}".format(50 * "=", i, self._length, time.time() - self._start, msg)
        sys.stdout.write("\r{}\n".format(line_str))
pb = ProcessBar()

import tensorflow as tf
class TensorBoardLogger:
    def __init__(self, names, path):
        self._values = {}
        with tf.name_scope('epoch'):
            for name in names:
                self._values['epoch/{}'.format(name)] = 0.
        with tf.name_scope('valid'):
            for name in names:
                self._values['valid/{}'.format(name)] = 0.
        self._file_summary = tf.summary.FileWriter(path)
        self._update_times = 0
    def log(self, params, step):
        for name in params.keys():
            summary = tf.Summary(value = [tf.Summary.Value(tag='train/' + name, simple_value=params[name])])
            self._file_summary.add_summary(summary, step)
        self.update(params, 'epoch')
    def update(self, params, mode):
        for name in params.keys():
            self._values['{}/{}'.format(mode, name)] += params[name]
        self._update_times += 1
    def summary(self, mode, epoch):
        return_dict = {}
        steps = self._update_times
        for key in self._values.keys():
            if key.startswith(mode):
                if self._values[key] == 0.0:
                    continue
                value = self._values[key] / steps
                return_dict[key.replace(mode + "/", "")] = value
                self._values[key] = 0.
                summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
                self._file_summary.add_summary(summary, epoch)
        self._update_times = 0
        return return_dict
    def summary_params(self, mode, params, epoch):
        for name in params.keys():
            tag_name = "{}/{}".format(mode, name)
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag_name, simple_value=params[name])])
            self._file_summary.add_summary(summary, epoch)