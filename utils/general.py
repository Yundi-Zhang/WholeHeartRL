import numpy as np
import torch
import json
import medutils
import os
import matplotlib.pyplot as plt
import os
import json
from typing import Union


def normalize_image(im: Union[np.ndarray, torch.Tensor], low: float = None, high: float = None, clip=True, scale: float=None) -> Union[np.ndarray, torch.Tensor]:
    """ Normalize array to range [0, 1] """
    if low is None:
        low = im.min()
    if high is None:
        high = im.max()
    if clip:
        im = im.clip(low, high)
    im_ = (im - low) / (high - low)
    if scale is not None:
        im_ = im_ * scale
    return im_


def image_normalization(image, scale=1, mode="2D"):
    if isinstance(image, np.ndarray) and np.iscomplexobj(image):
        image = np.abs(image)
    low = image.min()
    high = image.max()
    im_ = (image - low) / (high - low)
    if scale is not None:
        im_ = im_ * scale
    return im_


def to_1hot(class_indices: torch.Tensor, num_class) -> torch.Tensor:
    """ Converts index array to 1-hot structure. """
    origin_shape = class_indices.shape
    class_indices_ = class_indices.view(-1, 1).squeeze(1)
    
    N = class_indices_.shape[0]
    seg = class_indices_.to(torch.long).reshape((-1,))
    seg_1hot_ = torch.zeros((N, num_class), dtype=torch.float32, device=class_indices_.device)
    seg_1hot_[torch.arange(0, seg.shape[0], dtype=torch.long), seg] = 1
    seg_1hot = seg_1hot_.reshape(*origin_shape, num_class)
    return seg_1hot
