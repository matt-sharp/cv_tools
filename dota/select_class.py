import os
import shutil

def get_class(file_name,class_name):
    with open(file_name,'r') as f:
        a_lines = f.readlines()
        new_lines = []
        for l_s in a_lines:
            l = l_s.strip().split(' ')
            if l[8]  in class_name:
                new_lines.append(l_s)
    return new_lines


def select_class(srcdir,desdir,class_name):
    label_files = os.listdir(srcdir)
    i = 0
    for l in label_files:
        label_path = os.path.join(srcdir,l)
        new_lines = get_class(label_path,class_name)
        if (len(new_lines)==0):
            continue
        i +=1
        print(i)
        des_path = os.path.join(desdir,l)
        with open(des_path,'w') as f:
            f.writelines(new_lines)
            
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
    class_name = ['storage-tank','bridge','ground-track-field','roundabout']
    # select_class('/home/zoucg/cv_project/data/dota/val_crop_1024/labelTxt','/home/zoucg/cv_project/data/others/val/labelTxt',class_name)
    cp_images('/home/zoucg/cv_project/data/dota/train_crop_1024/images','/home/zoucg/cv_project/data/vehicle/train/images','/home/zoucg/cv_project/data/vehicle/train/imglist')

            
