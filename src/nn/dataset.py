# Python Imports
import os
import re

# Library Imports
import torch
from torch.utils.data import Dataset
import numpy as np
import tifffile
from PIL import Image

# Local Imports
from src.utils import get_tiff_resolution, get_resolution

MAX_N_INST = 380


class PredictionDataset(Dataset):
    def __init__(self, _configs, _source_directory, _target_resolution, _sample_dimension, _steps):

        self.configs = _configs
        self.source_directory = _source_directory

        self.dimension = _sample_dimension
        self.steps = _steps

        self.target_resolution = _target_resolution

        self.image_files = []
        self.per_img = []

        directory_content = os.listdir(_source_directory)
        directory_content = list(filter(lambda x: re.match(r'(.+).(tiff|tif)', x), directory_content))
        for file_name in directory_content:
            self.image_files.append(file_name)
            sh = tifffile.imread(os.path.join(_source_directory, file_name)).shape
            res = get_tiff_resolution(os.path.join(_source_directory, file_name), sh[-1], _target_resolution)
            scale = round(res / self.target_resolution, 2)
            sh = sh[-1] * scale
            steps_per_axis = int((sh - _sample_dimension) // self.steps + 1)
            self.per_img.append(steps_per_axis ** 2)

        self.imgs = {}
        self.per_img_cumsum = np.cumsum(self.per_img)
        self.cur_img_i = -1

    def n_per_img(self, img_nb):
        return self.per_img[img_nb]

    def __len__(self):
        return np.sum(self.per_img)

    def n_imgs(self):
        return len(self.image_files)

    def read_file(self, fn):
        img = tifffile.imread(os.path.join(self.source_directory, fn))

        # Choosing the channel and max projecting if needed

        if self.configs['is_stacked']:
            if len(img.shape) > 3:
                if img.shape[0] == 2:
                    img = np.max(img, axis=1)
                else:
                    img = np.max(img, axis=0)
            if len(img.shape) == 3:
                img = np.max(img, axis=0)

        if len(img.shape) > 2:
            img = img[self.configs['target_channel']]

        img = Image.fromarray(img)
        w, h = img.size
        res = get_resolution(os.path.join(self.source_directory, fn), w)
        scale = round(res / self.target_resolution, 2)
        if scale != 1:
            newW, newH = int(scale * w), int(scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small'
            img = img.resize((newW, newH))

        img = np.array(img)
        img = img.astype(np.float32)
        img = img / np.max(img)
        img = np.expand_dims(img.astype(np.float32), 0)
        return img

    def image_shape_by_id(self, _image_id):
        fn = self.image_files[_image_id]
        return self.imgs[fn].shape

    def __getitem__(self, i):
        file_i = np.min(np.where(self.per_img_cumsum > i)[0])

        n = i
        if file_i > 0:
            n = i - self.per_img_cumsum[file_i - 1]

        fn = self.image_files[file_i]
        if not (fn in self.imgs):
            self.imgs[fn] = self.read_file(fn)

        d_img = self.imgs[fn].shape[1]
        steps_per_axis = (d_img - self.dimension) // self.steps + 1
        x = self.steps * (n // steps_per_axis)
        y = self.steps * (n % steps_per_axis)

        img = self.imgs[fn][:, x:(x + self.dimension), y:(y + self.dimension)]

        return {'image': torch.from_numpy(img),
                'offs': torch.from_numpy(np.array([file_i, x, y, d_img], dtype=np.int32))}
