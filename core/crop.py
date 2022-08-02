import math

import torch
import torchvision.transforms.functional as TF

import numpy as np
from math import ceil

from data.transforms import scale_and_center_crop
from configs.config import cfg
from data.dataloader import get_class_size
from core.utils import make_divisible


class BaseCrop:
    def __init__(self):
        self.bboxes = None
        # To rescale the cropped tensor to a certain size. The rescaled tensor will be fed into
        # the final classification model.
        self.scale_size = make_divisible(cfg.TRAIN.IM_SIZE * cfg.SCALING.RES_MULT_B, cfg.SCALING.ROUND)

    @staticmethod
    def crop_and_resize_imgs(imgs, bboxes, scale_size=0):
        cropped_imgs = []
        for idx, (img, bbox) in enumerate(zip(imgs, bboxes)):
            cropped_img = img.crop(bbox)
            if scale_size != 0:
                cropped_img = scale_and_center_crop(cropped_img, scale_size, scale_size)

            cropped_imgs.append(cropped_img)
        return cropped_imgs

    @staticmethod
    def scale_and_center_crop_tensor(tensor, crop_size, scale_size):
        h, w = tensor.size(-2), tensor.size(-1)

        if w < h and w != scale_size:
            w, h = scale_size, int(h / w * scale_size)
        elif h <= w and h != scale_size:
            w, h = int(w / h * scale_size), scale_size

        img = TF.resize(tensor, size=[h, w])
        x = ceil((w - crop_size) / 2)
        y = ceil((h - crop_size) / 2)
        return TF.crop(img, y, x, crop_size, crop_size)

    def crop_and_resize_tensor(self, batch_tensor, bboxes, scale_size):
        n = batch_tensor.size(0)
        assert n == len(bboxes), "Mismatch of the number of images and boxes. " \
                                 "Got {} and {}, respectively.".format(n, len(bboxes))
        new_batch_tensor = []

        for i in range(n):
            box = bboxes[i]
            left, top, h, w = box[0], box[1], box[3] - box[1], box[2] - box[0]
            tensor = torch.unsqueeze(batch_tensor[i], 0)
            new_tensor = TF.crop(tensor, top, left, h, w)
            if h == w:
                new_tensor = TF.resize(new_tensor, size=scale_size)
            else:
                new_tensor = self.scale_and_center_crop_tensor(new_tensor, scale_size, scale_size)
            new_batch_tensor.append(new_tensor)

        return torch.concat(new_batch_tensor, dim=0)

    @staticmethod
    def get_square_bbox(bbox, size):
        # h, w = cam.shape[0], cam.shape[1]
        # assert h == w, "Mismatch of the height and width of the cam. Got {} and {}, respectively.".format(h, w)
        # cam_size = h

        # xmin, ymin, xmax, ymax = self.get_bbox(cam)
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin

        if height == width:
            return bbox
        elif width < height:
            new_width = height
            left_padding = (new_width - width) // 2
            right_padding = left_padding + (new_width - width) % 2
            xmin = xmin - left_padding
            xmax = xmax + right_padding

            if xmin < 0:
                offset = -xmin
                xmin = 0
                xmax += offset
            elif xmax > size:
                offset = xmax - size
                xmin -= offset
                xmax = size
        else:
            new_height = width
            up_padding = (new_height - height) // 2
            down_padding = up_padding + (new_height - height) % 2
            ymin = ymin - up_padding
            ymax = ymax + down_padding

            if ymin < 0:
                offset = -ymin
                ymin = 0
                ymax += offset
            elif ymax > size:
                offset = ymax - size
                ymin -= offset
                ymax = size

        box = [xmin, ymin, xmax, ymax]
        return box


class LearnedCrop(BaseCrop):
    def __init__(self, model):
        super(LearnedCrop, self).__init__()
        self.model = model

    @staticmethod
    def larger_side(box):
        return max(box[2] - box[0], box[3] - box[1])

    @staticmethod
    def expand_box(box, new_size, inputs_size):
        xc = int((box[0] + box[2]) / 2)
        yc = int((box[1] + box[3]) / 2)
        xmin = xc - int(new_size / 2)
        xmax = xmin + new_size
        ymin = yc - int(new_size / 2)
        ymax = ymin + new_size

        # Check if the box is out of the boundary
        if xmin < 0:
            xmin = 0
            xmax = new_size
        if ymin < 0:
            ymin = 0
            ymax = new_size
        if xmax > inputs_size:
            xmax = inputs_size
            xmin = xmax - new_size
        if ymax > inputs_size:
            ymax = inputs_size
            ymin = ymax - new_size

        return [xmin, ymin, xmax, ymax]

    def regulate_boxes(self, input_size):
        sizes = [make_divisible(input_size * ratio, cfg.SCALING.ROUND) for ratio in cfg.CROP.BOX_SIZES]

        for idx, box in enumerate(self.bboxes):
            box_size = self.larger_side(box)
            # print("Box before: {}, Size before {}".format(box, np.sqrt((box[2] - box[0]) * (box[3] - box[1]))))
            for size in sizes:
                if box_size <= size:
                    self.bboxes[idx] = self.expand_box(box, size, input_size)
                    break
            # print("Box after: {}, Size after{}".format(self.bboxes[idx], np.sqrt((self.bboxes[idx][2] - self.bboxes[idx][0]) * (self.bboxes[idx][3] - self.bboxes[idx][1]))))

    def forward(self, input_tensor, targets, bboxes=None, tune_ids=None):
        input_size = input_tensor.size(-1)

        if bboxes is None:
            self.model.eval()
            with torch.no_grad():
                output = self.model(input_tensor)
                # The obtained boxes from learned cropper are in a normalized form between [0, 1]
                self.bboxes = output.cpu().numpy()
                # Scale up the 0-1 normalized boxes to the regular size.
                self.bboxes = [[round(e * input_size) for e in box] for box in self.bboxes]
        else:
            self.bboxes = bboxes

        if cfg.CROP.SHAPE == 'square':
            self.bboxes = [self.get_square_bbox(box, input_size) for box in self.bboxes]

        if cfg.CROP.REGULATE:
            self.regulate_boxes(input_size)

        return self.crop_and_resize_tensor(input_tensor, self.bboxes, self.scale_size)

    def __call__(self, input_tensor, targets, **kwargs):
        return self.forward(input_tensor, targets, **kwargs)


def return_crop(crop_type):
    # Supported crop method
    _CROPS = {
        'cropnet_anynet_light_box': LearnedCrop,
    }
    err_message = f"Cropping method {crop_type} not supported"
    assert crop_type in _CROPS.keys(), err_message

    return _CROPS[crop_type]


def arrange_bboxes(boxes, split):
    """Arrange the boxes from crop_save() into the layout of [class, imgs].
    The input boxes are in the layout of [iter, rank, imgs].
    """
    class_sizes = get_class_size(split)
    sorted_all_boxes = []
    class_boxes = []
    cur_class_id = 0

    for iter_boxes in boxes:
        nboxes_per_iter = int(len(iter_boxes) * len(iter_boxes[0]))
        for box_id in range(nboxes_per_iter):
            if cur_class_id >= len(class_sizes):
                # When the drop_last=False, Pytorch will add extra indices
                # to make the data evenly divisible across the replicas.
                # Here we discard the added samples.
                break

            cuda_id = box_id % 4
            # The box index in current cuda device.
            cur_cuda_box_id = box_id // 4
            box = iter_boxes[cuda_id][cur_cuda_box_id]
            class_boxes.append(box)

            if len(class_boxes) == class_sizes[cur_class_id]:
                sorted_all_boxes.append(class_boxes)
                class_boxes = []
                cur_class_id += 1

    return np.array(sorted_all_boxes, dtype=object)
