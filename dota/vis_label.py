import cv2
from glob import glob

def visual_label(label_file,image_file):
    img = cv2.imread(image_file)
    with open(label_file ,'r') as f:
        a = f.readlines()
        a = list(map(lambda x :x.strip().split()[2:],a))
        
        for i in a:
            img = cv2.rectangle(img,(int(float(i[0])),int(float(i[1]))),(int(float(i[2])),int(float(i[3]))),color=(255,0,0))
        # img = cv2.putText(img,l[0],(location[0],location[1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    cv2.imwrite('a.png',img)


label_file = 'airplane1.txt'
image_file = '/data1/test_airplane/beijing_airport1_2017.tif'

visual_label(label_file,image_file)