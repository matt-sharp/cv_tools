import cv2
import os
from glob import glob

def change(img_path,dest_path):
    img = cv2.imread(img_path,-1)
    if img.ndim==2:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    if img.ndim==3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        cv2.imwrite(dest_path,img)
        pass

def main():
    img_dir = '/home/zoucg/data/18'
    img_files = glob(img_dir+'/*tif')
    dest_dir = '/home/zoucg/data/18_gray'
    for i in img_files:
        dest_path = os.path.join(dest_dir,os.path.basename(i))
        print(i)
        change(i,dest_path)

if __name__=="__main__":
    main()