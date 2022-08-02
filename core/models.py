import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import json
import numpy as np

from networks.resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet50_D09,
    resnet50_D08,
    resnet50_D07,
    resnet50_D06,
    resnet50_D05,
    resnet50_D04,
    resnet50_D03,
    resnet101,
    resnet152,
    wide_resnet50_2,
    wide_resnet101_2
)
from networks.cropnet import (
    cropnet_anynet_light_box,
)
from networks.regnet import (
    regnet_x_1_6gf_d05,
    regnet_x_1_6gf_d06,
    regnet_x_1_6gf_d07,
    regnet_x_1_6gf_d08,
    regnet_x_1_6gf_d09,
    regnet_x_1_6gf,
)


from configs.config import cfg
from core.crop import return_crop
import core.logging as logging

logger = logging.get_logger(__name__)

# Supported networks
_models = {
    'resnet50': resnet50,
    'resnet50_D09': resnet50_D09,
    'resnet50_D08': resnet50_D08,
    'resnet50_D07': resnet50_D07,
    'resnet50_D06': resnet50_D06,
    'resnet50_D05': resnet50_D05,
    'regnet_x_1_6gf': regnet_x_1_6gf,
    'regnet_x_1_6gf_D09': regnet_x_1_6gf_d09,
    'regnet_x_1_6gf_D08': regnet_x_1_6gf_d08,
    'regnet_x_1_6gf_D07': regnet_x_1_6gf_d07,
    'regnet_x_1_6gf_D06': regnet_x_1_6gf_d06,
    'regnet_x_1_6gf_D05': regnet_x_1_6gf_d05,
    'cropnet_anynet_light_box': cropnet_anynet_light_box,
}


def get_model_type():
    types = ['backbone', 'cropper']
    model_type = cfg.MODEL.TYPE
    err_message = "Model type {} not supported.".format(model_type)
    assert model_type in types, err_message

    return model_type


def return_model(model_name):
    err_message = f"Model {model_name} not supported."
    assert model_name in _models.keys(), err_message
    return _models[model_name]


def load_params(path):
    file_type = path.split('.')[-1]
    if file_type == 'json':
        with open(path, 'r') as f:
            params = json.load(f)
    elif file_type == 'npy':
        params = list(np.load(path, allow_pickle=True))
    else:
        raise NotImplementedError("File type '{}' not supported.".format(file_type))

    return params


def get_target_layers(cam_model, name):
    target_layers = {
        'resnet18': [cam_model.layer4],
        'resnet50': [cam_model.layer4]
    }

    return target_layers[name]


def setup_crop_model(device):
    """Build the model that execute the cropping.
    """
    # Supported cropping methods
    crop_type = cfg.CROP.TYPE
    crop = return_crop(crop_type)

    if crop_type == 'camcrop':
        cam_type = cfg.CROP.CAM
        weights = cfg.MODEL.CAM_WEIGHTS
        cam_model_name = cfg.CROP.CAM_MODEL
        cam_model = return_model(cam_model_name)()

        if weights:
            ckpt = torch.load(weights)
            if ckpt['test_acc'] > ckpt['ema_acc']:
                cam_model.load_state_dict(ckpt['model_state'])
            else:
                cam_model.load_state_dict(ckpt['ema_state'])

        target_layers = get_target_layers(cam_model, cam_model_name)
        cam = return_cam(cam_type)

        if cam_type == 'gradcam':
            gradcam = cam(cam_model, target_layers, device)
            return crop(gradcam, cam_split=cfg.CROP.CAM_SPLIT)
        else:
            raise NotImplementedError(f'CAM {cam_type} not implemented.')
    elif crop_type == 'boxcrop':
        return crop()
    else:
        # Use learned cropping
        box_predictor = return_model(crop_type)()
        box_predictor = box_predictor.to(device)

        return crop(box_predictor)


def setup_tune_model(device):
    """Build the model that predict the tuning direction."""
    tune_type = cfg.CROP.TUNE_MODEL
    if not tune_type:
        return None

    tune_model = return_model(tune_type)()
    tune_model = tune_model.to(device)
    logger.info("Tune model:\n{}".format(tune_model)) if cfg.VERBOSE else ()

    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        tune_model = DDP(module=tune_model, device_ids=[device], output_device=device, broadcast_buffers=True)

    return tune_model


def setup_model(device):
    """Build model and move to current device.
    """
    model_name = cfg.MODEL.NAME
    model = return_model(model_name)()
    start_epoch = 0

    logger.info("Model:\n{}".format(model)) if cfg.VERBOSE else ()
    # logger.info(logging.dump_log_data(net.complexity(model), "complexity"))

    # Load locally saved pretrained weights
    # TODO: Fix the loading of local weights
    fpath = cfg.MODEL.WEIGHTS
    if fpath:
        logger.info(f"Load pretrained weights from {fpath}")
        ckpt = torch.load(fpath, map_location='cpu')
        if ckpt['test_acc'] > ckpt['ema_acc']:
            model.load_state_dict(ckpt['model_state'])
        else:
            model.load_state_dict(ckpt['ema_state'])
        start_epoch = ckpt['epoch']

    # Transfers model to current device
    model = model.to(device)

    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = DDP(module=model, device_ids=[device], output_device=device, broadcast_buffers=True)

    return model, start_epoch
