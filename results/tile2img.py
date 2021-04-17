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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('imgs_dir', default='', type=str, help='imgs_dir')
    parser.add_argument('--out', default='', type=str, help="ouput image")

    args = parser.parse_args(sys.argv[1:])

    if not os.path.isdir(args.imgs_dir) or not os.path.exists(args.imgs_dir):
        print('目录不存在: {}'.format(args.imgs_dir))
        sys.exit(-1)
    imgs_dir = os.path.abspath(args.imgs_dir) + os.path.sep

    image_out = args.out
    if not image_out:
        image_out = imgs_dir + 'result.jpg'

    rects = []
    rect_max = {'left': -1, 'top': -1, 'width': 0, 'height': 0, 'img': ''}
    files = os.listdir(imgs_dir)
    for line in files:
        line = line.strip('\n')
        m = re.match(r'.*subimg_(\d+)_(\d+)_(\d+)_(\d+)_(\d+(\.\d+)?)\.*', line)
        if m:
            rect = {'left': int(m.group(1)), 'top': int(m.group(2)), 'width': int(m.group(3)), 'height': int(m.group(4)), 'img': line}
            rects.append(rect)
            if rect['left'] >= rect_max['left'] and rect['top'] >= rect_max['top']:
                rect_max = rect

    if rects:
        print('找到{}个图片'.format(len(rects)))
    else:
        print('找不到图片')
        sys.exit(-1)

    img0 = cv2.imdecode(np.fromfile(imgs_dir + rects[0]['img'], dtype=np.uint8), -1)
    print(img0.shape)
    width = rect_max['left'] + rect_max['width']
    height = rect_max['top'] + rect_max['height']
    shape = (height, width) if len(img0.shape) < 3 else (height, width, img0.shape[2])
    image = np.zeros(shape, dtype=img0.dtype)
    print('输出图片{}x{}'.format(shape[1], shape[0]))

    for rect in rects:
        img = cv2.imdecode(np.fromfile(imgs_dir + rect['img'], dtype=np.uint8), -1)
        image[rect['top']:rect['top'] + rect['height'], rect['left']:rect['left'] + rect['width']] = img

    print('保存图片: {}'.format(image_out))
    cv2.imencode(os.path.splitext(image_out)[1], image)[1].tofile(image_out)

    sys.exit(0)

