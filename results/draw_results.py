import cv2
import os
from os.path import *
from glob import glob

def  draw_withtxt(txt_path,source_img_path,dest_img_path):
    with open(txt_path,'r') as f:
        lines = f.readlines()
        img = cv2.imread(source_img_path)
        for l in lines:
            l = l.strip().split(' ')
            ifs = lambda x: int(float(x))    

            location = list(map(ifs,l[2:]))
            img = cv2.rectangle(img,(location[0],location[1]),(location[2],location[3]),color=(255,0,0))
            img = cv2.putText(img,l[0],(location[0],location[1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        cv2.imwrite(dest_img_path,img)

def process0():
    txt_dir = '/home/zoucg/cv_project/mytools/temp/test_out_gray'
    txt_file = glob(txt_dir+'/*txt1')
    source_img_dir = '/home/zoucg/data/18_gray'
    dest_dir = '/home/zoucg/cv_project/mytools/temp/gray'
    if not os.path.exists(dest_dir):

        os.makedirs(dest_dir)
    from tqdm import tqdm
    for i in tqdm(txt_file):
        img_name = basename(i)[:-4]+'tif'
        source_img_path  = join(source_img_dir,img_name)
        dest_img_path = join(dest_dir,img_name)
        try:
            draw_withtxt(i,source_img_path,dest_img_path)
        except:
            pass

def process1():
    txt_dir = '/home/zoucg/cv_project/mytools/temp/test_out_rgb'
    txt_file = glob(txt_dir+'/*txt1')
    source_img_dir = '/home/zoucg/data/18'
    dest_dir = '/home/zoucg/cv_project/mytools/temp/rgb'
    if not os.path.exists(dest_dir):

        os.makedirs(dest_dir)
    from tqdm import tqdm
    for i in tqdm(txt_file):
        img_name = basename(i)[:-4]+'tif'
        source_img_path  = join(source_img_dir,img_name)
        dest_img_path = join(dest_dir,img_name)
        try:
            draw_withtxt(i,source_img_path,dest_img_path)
        except:
            pass

def main():
    from multiprocessing import  Process

    p0 = Process(target=process0)
    p1 = Process(target=process1)
    p0.start()
    p1.start()


if __name__=='__main__':
    # txt_path = '/home/zoucg/cv_project/yolov3/test_out1208/japan2_20161231_18.txt1'
    # s ='/home/zoucg/data/18/japan2_20161231_18.tif'
    # dest='./te.tif'
    # draw_withtxt(txt_path,s,dest)
    main()
           