# -*- coding: UTF-8 -*-
#
# tiles
#
# Copyright (c) 2018 by 8803530@qq.com.  All rights reserved.
#

import cv2
import sys
import os
import re
import argparse
import numpy as np
from skimage.io import imread, imsave
import skimage

class Param:
    def __init__(self):
        self.img = None
        self.outdir = None
        self.resize_f = None;
        self.ext = None


def save_sub_img(param, left, top, img_size_x, img_size_y):
    img = srcImg[top:top + img_size_y, left:left + img_size_x, 0: 3]
    filename = '{}subimg_{}_{}_{}_{}_{}.{}'.format(param.outdir, left, top, img_size_x, img_size_y, param.resize_f, param.ext)
    print('save {}'.format(filename))
    # cv2.imwrite(filename, img)
    imsave(filename, img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', default=False, type=bool, help='output list file')
    parser.add_argument('--outdir', default='', type=str, help='output directory')
    parser.add_argument('--ext', default='jpg', type=str, help="output image ext")
    parser.add_argument('--size', default=450, type=int, help='sub image size')
    parser.add_argument('--overlap', default=30, type=int, help='overlap')

    parser.add_argument('--debug', default=False, type=bool, help='')
    parser.add_argument('--resize_f', default=0.0, type=float, help='缩放')
    parser.add_argument('--resize_interpolation', default=1, type=int, help='缩放')
    parser.add_argument('--progress_begin', default=0, type=int, help='progress-begin')
    parser.add_argument('--progress_end', default=0, type=int, help='progress-end')
    parser.add_argument('--progress_step', default=1, type=int, help='progress-step')
    parser.add_argument('image', default='', type=str, help='input image')

    args = parser.parse_args(sys.argv[1:])
    if args.resize_f > 0:
        print('暂不支持的参数: resize_f')
        sys.exit(-1)
    if not os.path.isfile(args.image):
        print('文件不存在: {}'.format(args.image))
        sys.exit(-1)
    if len(args.outdir) > 0 and not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    img_size = args.size
    img_overlap = args.overlap
    assert img_size > img_overlap

    srcImg = imread(args.image)
    #srcImg = cv2.imread(args.image, -1)
    #if not srcImg:
    #    print('文件打开失败: {}'.format(args.image))
    #    sys.exit(-1)
    print(srcImg.shape)

    param = Param()
    param.img = srcImg
    param.outdir = os.path.abspath(args.outdir) + os.path.sep
    param.resize_f = args.resize_f
    param.ext = args.ext

    img_width = srcImg.shape[1]
    img_height = srcImg.shape[0]
    #img_width = 450 + 30
    #img_height = 450  + 30

    img_size_y = img_size
    img_size_x = img_size
    img_overlap_y = img_overlap
    img_overlap_x = img_overlap
    if img_height < img_size:
        img_size_y = img_height
        img_overlap_y = 0
    if img_width < img_size:
        img_size_x = img_width
        img_overlap_x = 0

    y_cnt = (img_height - img_size_y) // (img_size - img_overlap)
    x_cnt = (img_width - img_size_x) // (img_size - img_overlap)
    print(x_cnt, ' ', y_cnt)

    img_tail_x = img_width - img_size_x - x_cnt * (img_size - img_overlap)
    img_tail_y = img_height - img_size_y - y_cnt * (img_size - img_overlap)
    print(img_tail_x, ' ', img_tail_y)

    y = 0
    last_y = img_tail_y > 0
    last_one = False
    while True:
        top = y * (img_size - img_overlap)
        if last_one:
            top = img_height - img_size
        for x in range(x_cnt + 1):
            left = x * (img_size_x - img_overlap_x)
            print('{:5}, {:5}, {}, {}'.format(left, top, img_size_x, img_size_y))
            save_sub_img(param, left, top, img_size_x, img_size_y)
        if img_tail_x > 0:
            print('{:5}, {:5}, {}, {}'.format(img_width - img_size, top, img_size_x, img_size_y))
            save_sub_img(param, left, top, img_size_x, img_size_y)
        y += 1
        if y > y_cnt:
            if last_y:
                last_y = False
                last_one = True
                continue
            else:
                break
