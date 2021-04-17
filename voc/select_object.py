import os
import shutil
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
from glob import glob
import random
random.seed(0)
def select(label_dir):
    lable_files = glob(label_dir +'/*xml')
    name = []
    for i in lable_files:
        tree=ET.parse(i)
        root = tree.getroot()
        if root.find('object'):
            n = i.split('/')[-1][:-4]+'\n'
            name.append(n)
    a = random.sample(name,len(name)//10*2)
    with open('/data1/temp/withobject.txt','w') as f:
        f.writelines(name)


if __name__=='__main__':
    label_dir = '/data1/temp/VOC2020/Annotations'
    select(label_dir)

    