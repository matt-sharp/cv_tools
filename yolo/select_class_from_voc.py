import os
import shutil

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

def get_class(file_name,class_name):
    tree=ET.parse(file_name)
    root = tree.getroot()
    object = root.iter('object')
    for obj in object:
        clsname = obj.find('name').text
        if  clsname in class_name:
            print(clsname)
            return root.find('filename').text

    
def select_label(srcdir,dest_file,class_name):
    label_file =  os.listdir(srcdir)
   
    file_names = []
    for l in label_file:
        label_path = os.path.join(srcdir,l)
        file_name = get_class(label_path,class_name)
        if file_name is None:
            continue
        file_names.append(file_name[:-4]+'\n')
    print(len(file_names))
    with open(dest_file,'w') as f:
        f.writelines(file_names)

            
def cp_images(srcdir,desdir,file_list):
    with open(file_list,'r') as f:
        lines = f.readlines()
        lines = map(lambda x: x.strip(),lines)
        lines = list(lines)
        img = os.listdir(srcdir)
        count =0
        for i in  img:
            count +=1
            print(count)
            if i[:-4] in lines:
                img_path = os.path.join(srcdir,i)
                des_path  = os.path.join(desdir,i)
                shutil.copy(img_path,des_path)
    
if __name__=='__main__':
    # class_name = ['storage-tank','bridge','plane', 'ship','ground-track-field', 'roundabout','small-vehicle', 'large-vehicle']
    class_name = ['airport']
    select_label(r'D:\detect_35\DIOR\Annotations',r'D:\detect_35\DIOR\airport',class_name)
    # cp_images('/home/zoucg/cv_project/data/dota/val_crop_1024/images','/home/zoucg/cv_project/data/a_35/val/images','/home/zoucg/cv_project/data/a_35/val/imglist')

            
