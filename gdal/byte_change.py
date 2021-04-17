import numpy
from skimage import io

def image_change(image_path,des_path):
    im_d = io.imread(image_path)
    im_ds = im_d*255
    io.imsave(des_path,im_ds)



if __name__=='__main__':
    image_path = '../bud__2048__2048.tif'
    dest_path = '../results/bud_2048.tif'
    image_change(image_path,dest_path)