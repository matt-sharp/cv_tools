#!/usr/bin/env python3

from pathlib import Path
import cv2
from osgeo import gdal
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import os

from boxes import np_half_iou
from pascal_voc_io import PascalVocWriter, PascalVocReader


class SplitBase():
    def __init__(self,
                 basepath,
                 outpath,
                 patch_size=1024,
                 overlap_ratio=0.5,
                 iou_thresh=0.5,
                 num_workers=4,
                 year=2020,
                 verbose=True,
                 suffix=".tif"
                 ):
        self.basepath = Path(basepath)
        self.img_path = self.basepath / 'JPEGImages'
        self.anno_path = self.basepath / 'Annotations'
        self.train_sets = open(self.basepath / 'train.txt').read().strip().splitlines()
        self.val_sets = open(self.basepath / 'val.txt').read().strip().splitlines()

        self.outpath = Path(outpath)
        self.img_out_path, self.anno_out_path, self.set_out_path = self.mk_voc_dir(
            self.outpath, year)

        self.train_ids = []
        self.val_ids = []

        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio
        self.iou_thresh = iou_thresh
        self.suffix = suffix
        self.verbose = verbose

        self.slide_step = int(patch_size * (1 - overlap_ratio))

        self.num_workers = num_workers
        self.pool = Pool(num_workers)

    def mk_voc_dir(self, root, year=2019):
        if isinstance(root, str):
            root = Path(root)
        anno = root / "crop/Annotations"
        img = root / "crop/JPEGImages"
        sets = root / "crop"
        anno.mkdir(parents=True, exist_ok=True)
        img.mkdir(parents=True, exist_ok=True)
        sets.mkdir(parents=True, exist_ok=True)

        return img, anno, sets

    def read_window(self, img, left, top, xsize, ysize):
        band_count = img.RasterCount
        band_list = [img.GetRasterBand(band + 1) for band in range(band_count)]

        rgb_data = cv2.merge(
            [band.ReadAsArray(left, top, xsize, ysize) for band in band_list])
        return rgb_data[:, :, ::-1]  # RGB -> BGR

    def save_image_patch(self, img, patch_name, left, top, xsize, ysize):
        """save numpy images to disk

        Args:
            img (numpy.ndarray): numpy array contains images. CxHxW
            patch_name (str): file name without path and suffix
            suffix (str): image file suffix
        """
        # img = img.copy()
        # patch = img[top:top + self.patch_size, left:left + self.patch_size]
        patch = self.read_window(img, left, top, xsize, ysize)
        img_out = self.img_out_path / f"{patch_name}.jpg"
        cv2.imencode(".jpg", patch)[1].tofile(
            str(img_out))  # solve Chinese path problem

    def save_patches(self, img, objs, patch_name, left, top, right, down):
        # anno_out = self.anno_out_path / f"{patch_name}.xml"
        patch_box = np.array([left, top, right, down])[None, :]
        obj_boxes = np.array([obj["bbox"] for obj in objs])
        half_ious, inter_boxes = np_half_iou(obj_boxes, patch_box)

        half_ious = half_ious.squeeze()
        inter_boxes = inter_boxes.squeeze()

        width = right - left
        height = down - top
        xml_writer = PascalVocWriter(
            "TianzhiDataset", f'{patch_name}.jpg', (height, width, 3))
        for idx, half_iou in enumerate(half_ious):
            if half_iou >= self.iou_thresh:
                # print(patch_name, inter_boxes[idx])
                new_box = inter_boxes[idx] - np.array([left, top, left, top])
                new_box = new_box.astype("int")
                xml_writer.add_bbox(objs[idx]["name"], False, new_box.tolist())

        xml_writer.save(self.anno_out_path / f'{patch_name}.xml')
        self.save_image_patch(img, patch_name, left, top, width, height)

    def split_data(self):
        img_ids = [img.stem for img in self.img_path.glob(f"*{self.suffix}")]
        # pbar = tqdm(img_ids, desc="Spliting")

        if self.num_workers == 1:
            for img_id in img_ids:
                self.split_single_image(img_id)
        else:
            worker = partial(self.split_single_image)
            self.pool.map(worker, img_ids)

        # print(self.train_ids)
        # # save train.txt and val.txt
        # with (self.set_out_path/'train.txt').open('w') as fp:
        #     fp.write('\n'.join(self.train_ids))
        # with (self.set_out_path/'val.txt').open('w') as fp:
        #     fp.write('\n'.join(self.val_ids))

    def split_single_image(self, img_id):

        xml_file = self.anno_path / f"{img_id}.xml"
        objects = PascalVocReader(xml_file).get_shapes()
        img_file = self.img_path / f"{img_id}{self.suffix}"

        # img = cv2.imread(str(img_file))
        # img_height, img_width = img.shape[:2]
        img = gdal.Open(str(img_file))
        img_height = img.RasterYSize
        img_width = img.RasterXSize

        top, left = 0, 0
        # start_positions = []
        patch_ids = []
        while (top < img_height):

            # print(top)
            if (top + self.patch_size >= img_height):
                top = max(img_height - self.patch_size, 0)
            left = 0
            while (left < img_width):
                # print(f"left = {left}")
                if (left + self.patch_size >= img_width):
                    left = max(img_width - self.patch_size, 0)

                right = min(left + self.patch_size, img_width)
                down = min(top + self.patch_size, img_height)
                patch_name = f"{img_id}__{top}__{left}"
                self.save_patches(img, objects, patch_name,
                                  left, top, right, down)
                patch_ids.append(patch_name)

                if left + self.patch_size >= img_width:
                    break
                else:
                    left += self.slide_step
            if top + self.patch_size >= img_height:
                break
            else:
                top += self.slide_step

        
        img = None
        print(img_file)

        if img_id in self.train_sets:
            # self.train_ids += patch_ids
            with (self.set_out_path/'train.txt').open('a') as fp:
                fp.write('\n'.join(patch_ids) + '\n')
        else:
            # self.val_ids += patch_ids
            with (self.set_out_path/'val.txt').open('a') as fp:
                fp.write('\n'.join(patch_ids) + '\n')
        

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inp_voc_root', type=str,default='/data1/VOC2020',
                        help='orginal VOC dataset root directory')
    parser.add_argument('-o', '--output', type=str,default='/data1/VOC2020',
                        help='output VOC dataset dataset directory')

    args = parser.parse_args()

    splitbase = SplitBase(
        basepath=args.inp_voc_root,
        outpath=args.output,
        patch_size=1024,
        overlap_ratio=0.5,
        num_workers=1,
        suffix=".tif")
    splitbase.split_data()


# def split_imgae_and_label(imgdir,label_dir,dest_img_dir,dest_label_dir,name_list):
#     if not os.path.exists(dest_img_dir):
#         os.makedirs(dest_img_dir)
#     if not os.path.exists(dest_label_dir):
#         os.makedirs(dest_label_dir)


    



if __name__ == "__main__":
    import time
    s = time.time()
    main()
    print(time.time() - s)



