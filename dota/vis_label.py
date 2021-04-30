import cv2
from glob import glob
import os

def visual_label(label_file,image_file):
    img = cv2.imread(image_file)
    with open(label_file ,'r') as f:
        a = f.readlines()
        bbox = list(map(lambda x :x.strip().split()[0:8],a))
        
        for i in bbox:
             for i in range(3):
                cv2.line(img, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i+1) * 2], bbox[(i+1) * 2 + 1]), color=(255,0,0), thickness=2)
            cv2.line(img, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=(255,0,0), thickness=4)
            cv2.putText(img, '%s %.3f' % ('ship', 1), (bbox[0], bbox[1] + 10),
                        color=(255,0,0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
            # img = cv2.rectangle(img,(int(float(i[0])),int(float(i[1]))),(int(float(i[2])),int(float(i[3]))),color=(255,0,0))

        # img = cv2.putText(img,l[0],(location[0],location[1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    return img


def visual_all(img_dir,label_file_dir,dest_dir):
    images = glob.glob(img_dir+'/*png')
    if os.path.exists(dest_dir):
        os.path.makedirs(dest_dir)
    for i in images:
        name = os.path.basename(i)[:-4]
        label_file = label_file_dir+'name'+'.txt'
        img = visual_label(label_file,i)
        dest_file = os.path.join(dest_dir,name+'.png')
        cv2.imwrite(dest_file,img)

    


def main():
    img_dir = ''