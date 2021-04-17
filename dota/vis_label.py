import cv2
from glob import glob

def vis_all(lable_dir,image_dir,out_dir):
    lable_files = glob(label_dir+'*txt')
    