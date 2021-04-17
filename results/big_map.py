import cv2
import sys
import os
import re
import argparse
import numpy as np
import skimage.io
from skimage.io import imread, imsave
import detect_whole


class Param:
    def __init__(self):
        self.img = None
        self.outdir = None
        self.resize_f = '0.0'
        self.ext = None

def save_sub_img(srcImg,param, left, top, img_size_x, img_size_y):
    img = srcImg[top:top + img_size_y, left:left + img_size_x, 0: 3]
    filename = '{}subimg_{}_{}_{}_{}_{}.{}'.format(param.outdir, left, top, img_size_x, img_size_y, param.resize_f, param.ext)
    print('save {}'.format(filename))
    # cv2.imwrite(filename, img)
    imsave(filename, img)
    
def crop_image(img_path,output_dir,img_size, img_overlap=60):
    import time
    t0 = time.time()
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)
    ext = 'jpg'
    size = 512
    overlap = 30
    srcImg = imread(image_path)

    param = Param()
    param.img = srcImg
    param.outdir = output_dir+'/'
    param.ext = ext
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
            img = srcImg[top:top + img_size_y, left:left + img_size_x, 0: 3]
            filename = '{}subimg_{}_{}_{}_{}_{}.{}'.format(param.outdir, left, top, img_size_x, img_size_y, param.resize_f, param.ext)
            detect_whole.detect_whole(img,'weights/best.pt',filename)
            # save_sub_img(srcImg,param, left, top, img_size_x, img_size_y)
        if img_tail_x > 0:
            print('{:5}, {:5}, {}, {}'.format(img_width - img_size, top, img_size_x, img_size_y))
            # save_sub_img(srcImg,param, left, top, img_size_x, img_size_y)
            img = srcImg[top:top + img_size_y, left:left + img_size_x, 0: 3]
            filename = '{}subimg_{}_{}_{}_{}_{}.{}'.format(param.outdir, left, top, img_size_x, img_size_y, param.resize_f, param.ext)
            detect_whole.detect_whole(img,'weights/best.pt',filename)
        y += 1
        if y > y_cnt:
            if last_y:
                last_y = False
                last_one = True
                continue
            else:
                break

    t2 = time.time()-t0
    print("total time is {}".format(t2))
    # img = srcImg[top:top + img_size_y, left:left + img_size_x, 0: 3]
    # filename = '{}subimg_{}_{}_{}_{}_{}.{}'.format(param.outdir, left, top, img_size_x, img_size_y, param.resize_f, param.ext)
    # detect_whole.detect_whole(img,'weights/best.pt',filename)

def merge_image(image_dir,image_path):
    image_out = image_path
    rects = []
    rect_max = {'left': -1, 'top': -1, 'width': 0, 'height': 0, 'img': ''}
    files = os.listdir(image_dir)
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

    img0 = cv2.imdecode(np.fromfile(image_dir + rects[0]['img'], dtype=np.uint8), -1)
    print(img0.shape)
    width = rect_max['left'] + rect_max['width']
    height = rect_max['top'] + rect_max['height']
    shape = (height, width) if len(img0.shape) < 3 else (height, width, img0.shape[2])
    image = np.zeros(shape, dtype=img0.dtype)
    print('输出图片{}x{}'.format(shape[1], shape[0]))

    for rect in rects:
        img = cv2.imdecode(np.fromfile(image_dir + rect['img'], dtype=np.uint8), -1)
        image[rect['top']:rect['top'] + rect['height'], rect['left']:rect['left'] + rect['width']] = img

    print('保存图片: {}'.format(image_out))
    cv2.imencode(os.path.splitext(image_out)[1], image)[1].tofile(image_out)

    sys.exit(0)

if __name__=='__main__':
    
    image_path = './japan6_17.tif'
    crop_image(image_path,'./temp/',512)
    merge_image('./temp/','test.jpg')
    print('保存图片: {}'.format(image_out))
    cv2.imencode(os.path.splitext(image_out)[1], image)[1].tofile(image_out)

    # sys.exit(0)