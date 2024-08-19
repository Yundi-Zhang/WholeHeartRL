from math import e
from pathlib import Path
import torch
import pandas as pd
import pickle
import numpy as np
from torchvision import transforms as v2
from torch.utils.data import Dataset
from typing import Tuple, Optional

from utils import image_normalization
from data.transforms import *


__all__ = ["Cardiac2DplusT", "Cardiac2DplusT_Test", "Cardiac3DSAX", "Cardiac3DSAX_Test", 
           "Cardiac3DLAX", "Cardiac3DLAX_Test", "Cardiac3DplusTSAX", "Cardiac3DplusTSAX_Test", 
           "Cardiac3DplusTLAX", "Cardiac3DplusTLAX_Test", "Cardiac3DplusTAllAX", "Cardiac3DplusTAllAX_Test"]


class AbstractDataset(Dataset):
    def __init__(self, subject_paths: Tuple[Path], target_table: pd.DataFrame, target_value_name: list[int],
                 load_seg: bool = False,
                 augs: bool = True,
                 sax_slice_num: int = None,
                 num_classes: int = 4,
                 time_frame: int = 50,
                 **kwargs):
        self.subject_paths = subject_paths
        self.target_table = target_table
        self.target_value_name = target_value_name
        self.load_seg = load_seg
        self.augs = augs
        self.sax_slice_num = sax_slice_num
        self.num_classes = num_classes
        self.time_frame = time_frame
        self.z_seg_relative = kwargs.get("z_seg_relative", 4)
        self.augmentation = self._augment
        self.view = self.get_view()
        
    @property
    def _augment(self) -> bool:
        if self.load_seg:
            return False # Disable image augmentation for segmentation tast.
        else:
            return self.augs
    
    def get_view(self) -> int:
        raise NotImplementedError

    def __len__(self):
        return len(self.subject_paths)
    
    def __getitem__(self, index):
        raise NotImplementedError("__getitem__ is not implemented for AbstractDataset")


class AbstractDataset_Test(AbstractDataset):
    
    @property
    def _augment(self) -> bool:
        return False


class Cardiac2DplusT(AbstractDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slice_num = 1
        
    def __getitem__(self, index) -> Tuple[np.ndarray, int]:
        """Load image files in the form of 2D+t.
        
        images: [T, H, W] where S is the number of slices, T is the number of time frames, and X and Y are the spatial dimensions.
        age: int, which is the age of the subject in years.
        """
        subject_id = self.get_subject_id(index)
        im_data, seg_data = self.load_im_seg_arr(index, z_2D_random=True, 
                                                 z_seg_relative=self.z_seg_relative, z_num=self.slice_num)
        im_data = np.transpose(im_data, (2, 3, 0, 1))
        im_data = image_normalization(im_data)
        im_data = torch.from_numpy(im_data)
        if self.augmentation:
            im_data = self.apply_augmentations(im_data)
            
        if self.load_seg:
            seg_data = np.transpose(seg_data, (2, 3, 0, 1))
            seg_data = torch.from_numpy(seg_data)
            return im_data, seg_data, index
        else:
            target_value = self.load_values(subject_id)
            target_value = torch.from_numpy(target_value).reshape(1)
            return im_data, target_value, index

    def get_subject_id(self, index):
        return int(self.subject_paths[index].parent.name)
    
    def get_view(self) -> int:
        return 0 # For short-axis view #TODO: should include long-axis view
    
    def load_im_seg_arr(self, img_idx: int, 
                        z_2D_random: Optional[bool]=False,
                        z_seg_relative: Optional[int]=None,
                        z_num: Optional[int]=None,) -> Tuple[np.ndarray, np.ndarray]:
        """"Load short axis image and segmentation files in the form of 2D+t. 
        
        The slice is picked relative to the start of the LV. The output is in the form of [T, X, Y]. T is the number of time frames, X and Y are the spatial dimensions.
        :param img_idx: Index of image in dataset image list.
        :param z_seg_relative: Picked SAX slice relative to the start of the LV. If None, all slices are returned.
        :param z_num: Number of slices to return. If None, all slices are returned.
        """
        npy_path = self.subject_paths[img_idx]
        assert os.path.exists(npy_path), f"File not found: {npy_path}"
        if npy_path.name[-4:] == ".npy":
            process_npy = np.load(npy_path, allow_pickle=True).item()
        elif npy_path.name[-4:] == ".npz":
            process_npy = np.load(npy_path)
            
        sax_im_data = process_npy["sax"].astype(np.float32) # [H, W, S, T]
        lax_im_data = process_npy["lax"].astype(np.float32) # [H, W, S, T]
        seg_sax_data = process_npy["seg_sax"].astype(np.int32) # [H, W, S, T]
        seg_lax_data = process_npy["seg_lax"].astype(np.int32) # [H, W, 3, T], for 2ch, 3ch, and 4ch
        
        # Change segmentation labels of long-axis slices
        if self.slice_num != 3:
            seg_lax_data[seg_lax_data == 1] = 4
            seg_lax_data[seg_lax_data == 2] = 5
        # Pick a random 2D+t slice among the stack of multi-view.
        if z_2D_random:
            im_data = np.concatenate([lax_im_data, sax_im_data], axis=2)
            seg_data = np.concatenate([seg_lax_data, seg_sax_data], axis=2)
            
            z = np.random.randint(im_data.shape[2])
            slice_im_data = np.moveaxis(im_data[..., z : z + 1, :], 1, 0)
            slice_seg_data = np.moveaxis(seg_data[..., z : z + 1, :], 1, 0)
            return slice_im_data, slice_seg_data
        # Remove slice dimension and keep only 2D+time
        if z_seg_relative is not None:
            z_seg_start = (seg_sax_data[..., 0] == 1).any((0, 1)).argmax()
            z = z_seg_start + z_seg_relative
            seg_sax_data = seg_sax_data[:, :, z] # [H, W, T]
            sax_im_data = sax_im_data[:, :, z]
            assert len(sax_im_data.shape) == 3, f"Img path: {npy_path}"
        # Select only the from 3rd to 3+z_num slices with segmentation map
        if z_num is not None:
            z_seg_start = (seg_sax_data[..., 0] == 1).any((0, 1)).argmax() + 2
            z_max = min(z_seg_start + z_num, seg_sax_data.shape[-2])
            z_min = z_max - z_num
            seg_sax_data = seg_sax_data[..., z_min : z_max, :] # [H, W, z_num, T]
            sax_im_data = sax_im_data[..., z_min : z_max, :]
            assert sax_im_data.shape[-2] == z_num, f"Img path: {npy_path}, shape: {sax_im_data.shape}"
        # Mirror image on x=y so the RV is pointing left (standard short axis view)
        sax_im_data = np.moveaxis(sax_im_data, 1, 0)
        lax_im_data = np.moveaxis(lax_im_data, 1, 0)
        seg_sax_data = np.moveaxis(seg_sax_data, 1, 0)
        seg_lax_data = np.moveaxis(seg_lax_data, 1, 0)
        return sax_im_data, seg_sax_data, lax_im_data, seg_lax_data
    
    def load_values(self, subject_idx: int):
        """Load values from csv file."""
        target_value = self.target_table[self.target_table["eid_87802"] == subject_idx][self.target_value_name]
        target_value = np.array(target_value.iloc[0].tolist(), dtype=np.float32)
        return target_value
    
    def apply_augmentations(self, im):
        """Transform image using torchvision transforms.
        input: [..., T, H, W]
        output: [..., T, H, W]"""
        transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=180),
        ])
        im = transforms(im)
        contrast = v2.RandomAutocontrast(p=0.5)
        im = contrast(im.unsqueeze(-3))
        im = im.squeeze(-3) # Remove channel dimension
        return im

class Cardiac2DplusT_Test(Cardiac2DplusT):
        
    @property
    def _augment(self) -> bool:
        return False


class Cardiac3DplusTSAX(Cardiac2DplusT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slice_num = self.sax_slice_num

    def __getitem__(self, index):
        """Load short axis image files in the form of 3D+t.
        
        images: [S, T, H, W] where S is the number of slices, T is the number of time frames, and X and Y are the spatial dimensions.
        target_value: int, which is the target value of the subject.
        """
        subject_id = self.get_subject_id(index)
        im_data, seg_data, *_ = self.load_im_seg_arr(index, z_num=self.slice_num)
        im_data = np.transpose(im_data, (2, 3, 0, 1)) # Move the slice and time dimension to the front
        assert len(im_data.shape) == 4 and im_data.shape[0] == self.slice_num
        im_data = image_normalization(im_data)
        im_data = torch.from_numpy(im_data)
        if self.augmentation:
            im_data = self.apply_augmentations(im_data)
        if self.load_seg:
            seg_data = np.transpose(seg_data, (2, 3, 0, 1))
            seg_data = torch.from_numpy(seg_data)
            return im_data, seg_data, index
        else:
            target_value = self.load_values(subject_id)
            target_value = torch.from_numpy(target_value).reshape(1)
            return im_data, target_value, index

    def get_view(self) -> int:
        return 0 # For short-axis view
    

class Cardiac3DplusTSAX_Test(Cardiac3DplusTSAX):
    
    @property
    def _augment(self) -> bool:
        return False
    

class Cardiac3DSAX(Cardiac3DplusTSAX):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slice_num = self.sax_slice_num
        self.frame_idx = kwargs.get("frame_idx", 25)

    def __getitem__(self, index):
        """Load short axis image files in the form of 3D+t.
        
        images: [S, H, W] where S is the number of slices, and X and Y are the spatial dimensions.
        target_value: int, which is the target value of the subject.
        """
        subject_id = self.get_subject_id(index)
        im_data, seg_data, *_ = self.load_im_seg_arr(index, z_num=self.slice_num)
        im_data = np.transpose(im_data, (2, 3, 0, 1)) # Move the slice and time dimension to the front
        assert len(im_data.shape) == 4 and im_data.shape[0] == self.slice_num
        im_data = im_data[:, self.frame_idx, ...]
        im_data = image_normalization(im_data)
        im_data = torch.from_numpy(im_data)

        if self.augmentation:
            im_data = self.apply_augmentations(im_data)
        if self.load_seg:
            seg_data = np.transpose(seg_data, (2, 3, 0, 1))
            seg_data = seg_data[:, self.frame_idx, ...]
            seg_data = torch.from_numpy(seg_data)
            return im_data, seg_data, index
        else:
            target_value = self.load_values(subject_id)
            target_value = torch.from_numpy(target_value).reshape(1)
            return im_data, target_value, index
    
    def get_view(self) -> int:
        return 0 # For short-axis view

class Cardiac3DSAX_Test(Cardiac3DSAX):
    
    @property
    def _augment(self) -> bool:
        return False
    
    
class Cardiac3DplusTLAX(Cardiac2DplusT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slice_num = 3
        self.num_classes = 3

    def __getitem__(self, index):
        """Load long axis image files in the form of 3D+t.
        
        images: [S, T, H, W] where S is the number of slices, T is the number of time frames, and X and Y are the spatial dimensions.
        target_value: int, which is the target value of the subject.
        """
        # Load long axis images
        subject_id = self.get_subject_id(index)
        _, _, im_data, seg_data = self.load_im_seg_arr(index)
        im_data = np.transpose(im_data, (2, 3, 0, 1)) # Move the slice and time dimension to the front
        assert len(im_data.shape) == 4 and im_data.shape[0] == self.slice_num
        im_data = image_normalization(im_data)
        im_data = torch.from_numpy(im_data)
        if self.augmentation:
            im_data = self.apply_augmentations(im_data)
        if self.load_seg:
            seg_data = np.transpose(seg_data, (2, 3, 0, 1))
            seg_data = torch.from_numpy(seg_data)
            return im_data, seg_data, index
        else:
            target_value = self.load_values(subject_id)
            target_value = torch.from_numpy(target_value).reshape(1)
            return im_data, target_value, index
    
    def get_view(self) -> int:
        return 1 # For long-axis view

class Cardiac3DplusTLAX_Test(Cardiac3DplusTLAX):
    
    @property
    def _augment(self) -> bool:
        return False
    

class Cardiac3DLAX(Cardiac3DplusTLAX):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slice_num = 3
        self.num_classes = 3
        self.frame_idx = kwargs.get("frame_idx", 25)

    def __getitem__(self, index):
        """Load short axis image files in the form of 3D+t.
        
        images: [S, H, W] where S is the number of slices, and X and Y are the spatial dimensions.
        target_value: int, which is the target value of the subject.
        """
        # Load long axis images
        subject_id = self.get_subject_id(index)
        _, _, im_data, seg_data = self.load_im_seg_arr(index)
        im_data = np.transpose(im_data, (2, 3, 0, 1)) # Move the slice and time dimension to the front
        assert len(im_data.shape) == 4 and im_data.shape[0] == self.slice_num
        im_data = im_data[:, self.frame_idx, ...]
        im_data = image_normalization(im_data)
        im_data = torch.from_numpy(im_data)
        if self.augmentation:
            im_data = self.apply_augmentations(im_data)
        if self.load_seg:
            seg_data = np.transpose(seg_data, (2, 3, 0, 1))
            seg_data = seg_data[:, self.frame_idx, ...]
            seg_data = torch.from_numpy(seg_data)
            return im_data, seg_data, index
        else:
            target_value = self.load_values(subject_id)
            target_value = torch.from_numpy(target_value).reshape(1)
            return im_data, target_value, index
    
    def get_view(self) -> int:
        return 1 # For long-axis view


class Cardiac3DLAX_Test(Cardiac3DLAX):
    
    @property
    def _augment(self) -> bool:
        return False    
    

class Cardiac3DplusTAllAX(Cardiac2DplusT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = 6
        self.slice_num = self.sax_slice_num + 3

    def __getitem__(self, index):
        """Load image files in the form of 3D+t.
        
        images: [S, T, H, W] where S is the number of slices, T is the number of time frames, and X and Y are the spatial dimensions.
        target_value: int, which is the target value of the subject.
        """
        # Load short axis and long axis images
        subject_id = self.get_subject_id(index)
        sax_slice_num = self.slice_num-3
        sax_im_data, seg_sax_data, lax_im_data, seg_lax_data = self.load_im_seg_arr(index, z_num=sax_slice_num)
        
        im_data = np.concatenate([lax_im_data, sax_im_data], axis=2)
        im_data = np.transpose(im_data, (2, 3, 0, 1)) # Move the slice and time dimension to the front
        assert len(im_data.shape) == 4 and im_data.shape[0] == self.slice_num
        im_data = image_normalization(im_data)
        im_data = torch.from_numpy(im_data)
        if self.augmentation:
            im_data = self.apply_augmentations(im_data)
        if self.load_seg:
            # Relabel the long-axis segmentation
            seg_lax_data[..., 0, :][seg_lax_data[..., 0, :] == 1] = 4
            seg_lax_data[..., 2, :][seg_lax_data[..., 2, :] == 1] = 4
            seg_lax_data[..., 2, :][seg_lax_data[..., 2, :] == 2] = 5
            seg_data = np.concatenate([seg_lax_data, seg_sax_data], axis=2)
            seg_data = np.transpose(seg_data, (2, 3, 0, 1))
            seg_data = torch.from_numpy(seg_data)
            return im_data, seg_data, index
        else:
            target_value = self.load_values(subject_id)
            target_value = torch.from_numpy(target_value).reshape(1)
            return im_data, target_value, index
    
    def get_view(self) -> int:
        return 2 # For both long and short-axis views
    
    
class Cardiac3DplusTAllAX_Test(Cardiac3DplusTAllAX):
    
    @property
    def _augment(self) -> bool:
        return False
