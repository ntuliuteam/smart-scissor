import torch
import torch.backends.cudnn as cudnn

import sys
import os
import numpy as np
import random
from copy import deepcopy

from configs import config
from configs.config import cfg
from core.models import setup_model, setup_crop_model, get_model_type

import core.distributed as dist
import core.meters as meters
import data.dataloader as dataloader
import core.logging as logging
import data.transforms as transforms

try:
    import torch.cuda.amp as amp
except ImportError:
    amp = None


logger = logging.get_logger(__name__)


grad_blobs = []
def grad_hook(module, input, output):
    if not hasattr(output, "requires_grad") or not output.requires_grad:
        return

    def _save_grad(grad):
        grad = torch.squeeze(grad)
        #         print('Grad shape：', grad.shape)
        grad_blobs.append(grad.data.cpu().numpy())

    output.register_hook(_save_grad)


def setup_env():
    """Sets up environment for training or testing."""
    if dist.is_main_proc():
        # Ensure that the output dir exists
        outdir = cfg.OUT_DIR
        if not os.path.isdir(outdir):
            os.makedirs(cfg.OUT_DIR)
        # Save the config
        config.dump_cfg()
    # Setup logging
    logging.setup_logging()
    # Log torch, cuda, and cudnn versions
    version = [torch.__version__, torch.version.cuda, torch.backends.cudnn.version()]
    logger.info("PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    env = "".join([f"{key}: {value}\n" for key, value in sorted(os.environ.items())])
    logger.info(f"os.environ:\n{env}")
    # Log the config as both human readable and as a json
    logger.info("Config:\n{}".format(cfg)) if cfg.VERBOSE else ()
    logger.info(logging.dump_log_data(cfg, "cfg", None))
    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    cudnn.benchmark = cfg.CUDNN.BENCHMARK


def eval_model():
    setup_env()
    device = dist.get_device()
    model, start_epoch = setup_model(device)
    test_loader = dataloader.construct_test_loader()
    test_meter = meters.TestMeter(len(test_loader))
    test_epoch(model, test_loader, test_meter, 0, device)


def crop_eval():
    setup_env()
    device = dist.get_device()
    model, start_epoch = setup_model(device)
    crop_model = setup_crop_model(device)
    crop_test_loader = dataloader.construct_test_loader()
    test_meter = meters.TestMeter(len(crop_test_loader))
    crop_test_epoch(model, crop_model, crop_test_loader, test_meter, 0, device)


def crop_test_epoch(model, crop_model, loader, meter, cur_epoch, device):
    model.eval()
    meter.reset()
    meter.iter_tic()

    for cur_iter, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        cropped_inputs = crop_model(inputs, labels)

        # print(cropped_inputs.shape)

        with torch.no_grad():
            preds = model(cropped_inputs)
        acc1, acc5 = meters.topk_accuracy(preds, labels, (1, 5))
        acc1, acc5 = dist.scaled_all_reduce([acc1, acc5], cfg.NUM_GPUS)
        acc1, acc5 = acc1.item(), acc5.item()

        meter.iter_toc()
        meter.update_stats(acc1, acc5, inputs.size(0) * cfg.NUM_GPUS)
        meter.log_iter_stats(cur_epoch, cur_iter)
        meter.iter_tic()
    meter.log_epoch_stats(cur_epoch)


@torch.no_grad()
def test_epoch(model, loader, meter, cur_epoch, device):
    """Evaluates the backbone and cropper model on the test set."""
    # Enable eval mode
    model.eval()
    meter.reset()
    meter.iter_tic()

    accs = []

    for cur_iter, (inputs, labels) in enumerate(loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.to(device), labels.to(device)

        # Compute the predictions
        preds = model(inputs)
        # Compute the errors
        if cfg.MODEL.TYPE == 'backbone':
            # For testing image classification model
            top1_acc, top5_acc = meters.topk_accuracy(preds, labels, [1, 5])
        else:
            # For testing box predictor
            top1_acc = meters.get_miou(preds, labels)
            top5_acc = torch.tensor(0, dtype=torch.float32).cuda()

        # Combine the errors across the GPUs  (no reduction if 1 GPU used)
        top1_acc, top5_acc = dist.scaled_all_reduce([top1_acc, top5_acc], cfg.NUM_GPUS)
        # Copy the errors from GPU to CPU (sync point)
        top1_acc, top5_acc = top1_acc.item(), top5_acc.item()

        accs.append(top1_acc)

        meter.iter_toc()
        # Update and log stats
        meter.update_stats(top1_acc, top5_acc, inputs.size(0) * cfg.NUM_GPUS)
        meter.log_iter_stats(cur_epoch, cur_iter)
        meter.iter_tic()
    # Log epoch stats
    meter.log_epoch_stats(cur_epoch)
    np.save('eval_accs.npy', accs)


