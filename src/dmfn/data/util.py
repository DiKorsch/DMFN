import numpy as np
import torch
import random

from torchvision.transforms import functional as F
from PIL import Image

def tensor2img(image_tensor, imtype=np.uint8):
    images_numpy = []
    for i in range(len(image_tensor)):
        image_numpy = image_tensor[i].numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # [-1, 1] --> [0, 255]
        image_numpy = np.clip(image_numpy.round(), 0, 255).astype(imtype)
        images_numpy.append(image_numpy)
    return images_numpy


def im2tensor(image):
    img = Image.fromarray(image)
    img_tensor = F.to_tensor(img).float()
    return img_tensor


def bbox(opt):
    """Generate a center/random tlhw with configuration.

    Args:
        opt: opt should have configuration including
        vertical_margin, height, horizontal_margin, width.

    Returns:
        tuple: (top, left, height, width)

    """
    fineSize = opt['fineSize']
    if opt['mask_pos'] == 'random':  # random mask
        maxt = fineSize - opt['vertical_margin'] - opt['mask_height']  # max top
        maxl = fineSize - opt['horizontal_margin'] - opt['mask_width']  # max left
        top = random.randint(opt['vertical_margin'], maxt)
        left = random.randint(opt['horizontal_margin'], maxl)
    else:  # center mask
        top = (fineSize - opt['vertical_margin'] - opt['mask_height']) // 2
        left = (fineSize - opt['horizontal_margin'] - opt['mask_width']) // 2
    height = opt['mask_height']
    width = opt['mask_width']
    return (top, left, height, width)


def bbox2mask(bbox, opt):
    """Generative mask tensor from bbox.

    Args:
        bbox: configuration tuple, (top, left, height, width)
        opt: opt should have configuration including img_shapes,
        max_delta_height, max_delta_width.

    Returns:
        Tensor: output with shape [1, H, W]
    """

    def npmask(bbox, height, width, delta_h, delta_w):
        mask = np.zeros((1, height, width), np.float32)
        h = random.randint(0, delta_h // 2 + 1)  # [0, 16]
        w = random.randint(0, delta_w // 2 + 1)  # [0, 16]
        mask[:, bbox[0] + h:bbox[0] + bbox[2] - h,
        bbox[1] + w:bbox[1] + bbox[3] - w] = 1.
        return mask

    fineSize = opt['fineSize']
    mask = torch.from_numpy(npmask(bbox, fineSize, fineSize, opt['max_delta_height'], opt['max_delta_width']))
    return mask
