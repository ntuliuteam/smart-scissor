"""Meters."""

from collections import deque
import torchvision.ops as ops

import numpy as np
import time
import core.logging as logging
import torch
from configs.config import cfg


logger = logging.get_logger(__name__)


def topk_accuracy(output, targets, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k.
    """
    err_str = "Batch dim of predictions and labels must match"
    assert output.size(0) == targets.size(0), err_str

    with torch.no_grad():
        max_k = max(topk)
        batch_size = output.size(0)

        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def get_acc_meter():
    acc_meters = {
        'backbone': topk_accuracy
    }
    model_type = cfg.MODEL.TYPE
    err_message = f"Accuracy meter for {model_type} not supported."
    assert model_type in acc_meters.keys(), err_message

    return acc_meters[model_type]


def gpu_mem_usage():
    """Computes the GPU memory usage for the current device (MB)."""
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / 1024 / 1024


class ScalarMeter(object):
    """Measures a scalar value (adapted from Detectron)."""

    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        return np.median(self.deque)

    def get_win_avg(self):
        return np.mean(self.deque)

    def get_global_avg(self):
        return self.total / self.count


class Timer(object):
    """A simple timer (adapted from Detectron)."""

    def __init__(self):
        self.total_time = None
        self.calls = None
        self.start_time = None
        self.diff = None
        self.average_time = None
        self.reset()

    def tic(self):
        # using time.time as time.clock does not normalize for multithreading
        self.start_time = time.time()

    def toc(self):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls

    def reset(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0


class TestMeter(object):
    """Measures testing stats."""

    def __init__(self, epoch_iters, phase="test"):
        self.epoch_iters = epoch_iters
        self.phase = phase
        self.iter_timer = Timer()
        # Current minibatch accuracy (smoothed over a window)
        self.mb_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        # Max accuracy (over the full test set)
        self.max_top1_acc = 0.0
        self.max_top5_acc = 0.0
        # Number of correctly classified examples
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_samples = 0

    def reset(self, max_acc=False):
        if max_acc:
            self.max_top1_acc = 0.0
            self.max_top5_acc = 0.0
        self.iter_timer.reset()
        self.mb_top1_acc.reset()
        self.mb_top5_acc.reset()
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, top1_acc, top5_acc, mb_size):
        self.mb_top1_acc.add_value(top1_acc)
        self.mb_top5_acc.add_value(top5_acc)
        self.num_top1_cor += top1_acc * mb_size
        self.num_top5_cor += top5_acc * mb_size
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        mem_usage = gpu_mem_usage()
        iter_stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_avg": self.iter_timer.average_time,
            "time_diff": self.iter_timer.diff,
            "top1_acc": self.mb_top1_acc.get_win_avg(),
            "top5_acc": self.mb_top5_acc.get_win_avg(),
            "mem": int(np.ceil(mem_usage)),
        }
        return iter_stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.LOG_PERIOD == 0:
            stats = self.get_iter_stats(cur_epoch, cur_iter)
            logger.info(logging.dump_log_data(stats, self.phase + "_iter"))

    def get_epoch_stats(self, cur_epoch):
        top1_acc = self.num_top1_cor / self.num_samples
        top5_acc = self.num_top5_cor / self.num_samples
        self.max_top1_acc = max(self.max_top1_acc, top1_acc)
        self.max_top5_acc = max(self.max_top5_acc, top5_acc)
        mem_usage = gpu_mem_usage()
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "time_avg": self.iter_timer.average_time,
            "time_epoch": self.iter_timer.average_time * self.epoch_iters,
            "top1_acc": top1_acc,
            "top5_acc": top5_acc,
            "max_top1_acc": self.max_top1_acc,
            "max_top5_acc": self.max_top5_acc,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        logger.info(logging.dump_log_data(stats, self.phase + "_epoch"))


class CropMeter(object):
    """Measures cropping stats."""

    def __init__(self, epoch_iters, phase="crop"):
        self.epoch_iters = epoch_iters
        self.phase = phase
        self.iter_timer = Timer()
        # Current minibatch accuracy (smoothed over a window)
        self.num_samples = 0

    def reset(self):
        self.iter_timer.reset()
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, mb_size):
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        mem_usage = gpu_mem_usage()
        iter_stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_avg": self.iter_timer.average_time,
            "time_diff": self.iter_timer.diff,
            "mem": int(np.ceil(mem_usage)),
        }
        return iter_stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.LOG_PERIOD == 0:
            stats = self.get_iter_stats(cur_epoch, cur_iter)
            logger.info(logging.dump_log_data(stats, self.phase + "_iter"))

    def get_epoch_stats(self, cur_epoch):
        mem_usage = gpu_mem_usage()
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "time_avg": self.iter_timer.average_time,
            "time_epoch": self.iter_timer.average_time * self.epoch_iters,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        logger.info(logging.dump_log_data(stats, self.phase + "_epoch"))


def readable_time(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    mins = seconds % 3600 // 60
    secs = seconds % 60

    return '{}H-{}M-{}S'.format(hours, mins, secs)
