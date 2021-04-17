import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.modules.utils import _pair

from osgeo import gdal

from torchseg.utils import bytescale


class RSInferenceDataset(Dataset):
    def __init__(
            self,
            rs_file,
            patch_size=(512, 512),
            slide_step=(512, 512),
            to_rgb=True,
            pad=256,
            transform=None,
    ):
        super().__init__()
        self.rs_file = rs_file
        self.patch_size = _pair(patch_size)
        self.slide_step = _pair(slide_step)

        # get data info
        ds = gdal.Open(rs_file)
        self.data_info = self._get_data_info(ds)
        self.ids = self._get_patch_ids()
        ds = None

        self.to_rgb = to_rgb
        self.pad = pad
        # torchvision or albumentation
        self.transform = transform

    def __getitem__(self, idx):
        img = self._read_patch(idx)

        if self.transform:
            img = self.transform(img)

        return img, self.ids[idx]

    def __len__(self):
        return len(self.ids)

    def _get_data_info(self, src):
        return {
            'width': src.RasterXSize,
            'height': src.RasterYSize,
            'driver': src.GetDriver().ShortName,
            'dtype': gdal.GetDataTypeName(src.GetRasterBand(1).DataType),
            'bands': src.RasterCount,
            'proj': src.GetProjection(),
            'geotransform': src.GetGeoTransform(),
        }

    def _get_patch_ids(self):
        left, top = 0, 0
        width, height = self.data_info['width'], self.data_info['height']
        left_top_xy = []  # left-top corner coordinates (xmin, ymin)
        while left < width:
            if left + self.patch_size[0] >= width:
                left = max(width - self.patch_size[0], 0)
            top = 0
            while top < height:
                if top + self.patch_size[1] >= height:
                    top = max(height - self.patch_size[1], 0)
                # right = min(left + self.patch_size[0], width - 1)
                # bottom = min(top + self.patch_size[1], height - 1)
                # save
                left_top_xy.append((left, top))
                if top + self.patch_size[1] >= height:
                    break
                else:
                    top += self.slide_step[1]

            if left + self.patch_size[0] >= width:
                break
            else:
                left += self.slide_step[0]

        return left_top_xy

    def _read_patch(self, idx):
        xmin, ymin = self.ids[idx]
        xsize, ysize = self.patch_size
        pad_x = pad_y = self.pad

        xmin -= pad_x
        ymin -= pad_y
        left, top = 0, 0
        if xmin < 0:
            xmin = 0
            xsize += pad_x
            left = pad_x
        elif xmin + xsize + 2 * pad_x > self.width:
            xsize += pad_x
        else:
            xsize += 2 * pad_x

        if ymin < 0:
            ymin = 0
            ysize += pad_y
            top = pad_y
        elif ymin + ysize + 2 * pad_y > self.height:
            ysize += pad_y
        else:
            ysize += 2 * pad_y

        bands = self.data_info['bands']

        # to use multi-processing
        ds = gdal.Open(self.rs_file)
        ret = []
        for i in range(1, bands + 1):
            # band = self.ds.GetRasterBand(i)
            band = ds.GetRasterBand(i)
            img = np.zeros((self.patch_size[0] + 2 * pad_x,
                            self.patch_size[0] + 2 * pad_x))
            patch_img = band.ReadAsArray(
                xoff=xmin,
                yoff=ymin,
                win_xsize=xsize,
                win_ysize=ysize,
            )
            img[top:top + ysize, left:left + xsize] = patch_img

            ret.append(img)
        img = np.stack(ret, axis=-1)
        if self.to_rgb and bands == 4:
            img = bytescale(img)
            img = img[..., [2, 1, 0]]  # bgr -> rgb
        ds = None

        return img

    @property
    def width(self):
        return self.data_info['width']

    @property
    def height(self):
        return self.data_info['height']
