import os
import random
random.seed(0)
def reorg(orgin_txt,train_path,val_path):
    with open(orgin_txt,'r') as f:
        lines = f.readlines()
        val_lines = random.sample(lines,len(lines)//5)
        print(val_lines)
        trian_lines = list(set(lines)-set(val_lines))

    with open(train_path, 'w') as f1:
        f1.writelines(trian_lines)

    with open(val_path, 'w') as f2:
        f2.writelines(val_lines)

if __name__=='__main__':
    orgin_path = r'/data1/temp/voc_crop/VOC2020/train.txt'
    train_path = r'/data1/temp/voc_crop/VOC2020/train'
    val_path = r'/data1/temp/voc_crop/VOC2020/val'
    reorg(orgin_path, train_path,val_path)