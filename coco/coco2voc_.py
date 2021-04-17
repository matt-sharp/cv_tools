from pycocotools.coco import COCO
import os
import shutil
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
from tqdm import tqdm
import xml.dom.minidom


import os.path as osp
import cv2
import numpy as np
import random
from pycocotools.coco import COCO

# def get_label_map(label_file):
#     label_map = {}
#     labels = open(label_file, 'r')
#     for line in labels:
#         ids = line.split(',')
#         label_map[int(ids[0])] = int(ids[1])
#     return label_map

# class COCODetection():
#     def __init__(self,instances_path="coco/annotations/instances_train2017.json",
#                  imgdir_path="coco/train2017/"):
#         self.coco=COCO(annotation_file=instances_path)
#         self.root=imgdir_path
#         self.label_map=get_label_map('coco/coco_labels.txt')
#         self.ids=list(self.coco.imgToAnns.keys())  # 所有图像的ids

#     def get_img_target(self,index):
#         img_id=self.ids[index]
#         target = self.coco.imgToAnns[img_id]
#         ann_ids = self.coco.getAnnIds(imgIds=img_id)
#         target = self.coco.loadAnns(ann_ids)
#         path = osp.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])
#         assert osp.exists(path), 'Image path does not exist: {}'.format(path)
#         img = cv2.imread(osp.join(self.root, path))
#         height, width, _ = img.shape
#         scale = np.array([width, height, width, height])
#         res = []
#         for obj in target:
#             if 'bbox' in obj:
#                 bbox = obj['bbox']
#                 bbox[2] += bbox[0]
#                 bbox[3] += bbox[1]
#                 # label_idx为0留给background，也可以这里减一，最后80留给background
#                 label_idx = self.label_map[obj['category_id']]
#                 final_box = list(np.array(bbox) / scale)
#                 final_box.append(label_idx)
#                 res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
#             else:
#                 print("no bbox problem!")
#         return img,res
    
    
# if __name__=="__main__":
#     data_coco=COCODetection()
#     index=random.randint(0,len(data_coco.ids))
#     print(data_coco.get_img_target(index))
 
classes_names =['storage-tank','bridge','ground-track-field','roundabout']
def id2name(coco):
    classes=dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']]=cls['name']
    return classes
def showimg(coco,dataset,img,classes,cls_id,show=True):
    global dataDir
    I=Image.open('%s/%s/%s'%(dataDir,dataset,img['file_name']))
    #通过id，得到注释的信息
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    # print(annIds)
    anns = coco.loadAnns(annIds)
    # print(anns)
    # coco.showAnns(anns)
    objs = []
    for ann in anns:
        class_name=classes[ann['category_id']]
        if class_name in classes_names:
            print(class_name)
            if 'bbox' in ann:
                bbox=ann['bbox']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                obj = [class_name, xmin, ymin, xmax, ymax]
                objs.append(obj)
                draw = ImageDraw.Draw(I)
                draw.rectangle([xmin, ymin, xmax, ymax])
    if show:
        plt.figure()
        plt.axis('off')
        plt.imshow(I)
        plt.show()
 
    return objs


def prase_coco(label_path,xml_dir):
    if os.path.exists(xml_dir) is False:
        os.makedirs(xml_dir)

    coco = COCO(label_path)
    anns = coco.anns
    imgs = coco.imgs
    img_with_anns = coco.imgToAnns
    count = 0
    f = open('/home/zoucg/cv_project/data/a_35/train/temp.txt','w+')
    for key, value in img_with_anns.items():
        imgId = key
        # img = coco.loadImgs(imgId)[0]
        
        filename = imgs[imgId]['file_name']
        f.write(filename+'\n')
       
        img_height =imgs[imgId]['height'] 
        img_width = imgs[imgId]['width'] 
        objects =[]
        for v in value :
            bbox=v['bbox']
            category_id = int(v['category_id'])
            class_name = classes_names[category_id-1]

            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2] + bbox[0])
            ymax = int(bbox[3] + bbox[1])
            obj = [class_name, xmin, ymin, xmax, ymax]
            objects.append(obj)
        count = count+1
        print(count)
        dest_xml = os.path.join(xml_dir,filename[:-3]+'xml')
        labels = {'file_name':filename,'height':img_height,'width':img_width,'objects':objects}
        create_xml(dest_xml,labels)
        
        


def coco2xml(lable_path, dest_dir):
    pass

def create_xml(xml_path, label):
    im_height = label['height']
    im_width = label['width']

    doc = xml.dom.minidom.Document()
    root = doc.createElement('anotation')
    doc.appendChild(root)

    node_floder = doc.createElement('folder')
    node_floder.appendChild(doc.createTextNode('VOC2007'))

    node_filename = doc.createElement('filename')
    node_filename.appendChild(doc.createTextNode(label['file_name']))

    node_source = doc.createElement('source')
    node_database = doc.createElement('database')
    node_database.appendChild(doc.createTextNode('The VOC2007 Dotadataset'))
    node_anotation = doc.createElement('anotation')
    node_anotation.appendChild(doc.createTextNode('PASCAL VOC2007'))
    node_image = doc.createElement('image')
    node_image.appendChild(doc.createTextNode('fickrid'))
    node_fickrid = doc.createElement('fickrid')
    node_fickrid.appendChild(doc.createTextNode('341012865'))

    node_source.appendChild(node_database)
    node_source.appendChild(node_anotation)
    node_source.appendChild(node_image)
    node_source.appendChild(node_fickrid)

    node_ownner = doc.createElement('ownner')
    node_frick = doc.createElement('fickrid')
    node_frick.appendChild(doc.createTextNode('no'))
    node_name = doc.createElement('name')
    node_name.appendChild(doc.createTextNode('no'))
    node_ownner.appendChild(node_fickrid)
    node_ownner.appendChild(node_name)

    node_size = doc.createElement('size')
    node_width = doc.createElement('width')
    node_width.appendChild(doc.createTextNode(str(im_width)))
    node_height = doc.createElement('height')
    node_height.appendChild(doc.createTextNode(str(im_height)))
    node_depth = doc.createElement('depth')
    node_depth.appendChild(doc.createTextNode(str(3)))
    node_size.appendChild(node_width)
    node_size.appendChild(node_height)
    node_size.appendChild(node_depth)

    node_segment = doc.createElement('segmented')
    node_segment.appendChild(doc.createTextNode('0'))

    root.appendChild(node_floder)
    root.appendChild(node_filename)
    root.appendChild(node_source)
    root.appendChild(node_ownner)
    root.appendChild(node_size)
    root.appendChild(node_segment)

    for b in label['objects']:
        # print(b)
        node_object = doc.createElement('object')

        node_name1 = doc.createElement('name')
        node_name1.appendChild(doc.createTextNode(b[0]))
        node_pose = doc.createElement('pose')
        node_pose.appendChild(doc.createTextNode('left'))
        node_truncated = doc.createElement('truncated')
        node_truncated.appendChild(doc.createTextNode('0'))
        node_diffcult = doc.createElement('difficult')
        node_diffcult.appendChild(doc.createTextNode('0'))

        node_bndbox = doc.createElement('bndbox')
        node_x0 = doc.createElement('xmin')
        node_x0.appendChild(doc.createTextNode(str(b[1])))
        node_y0 = doc.createElement('ymin')
        node_y0.appendChild(doc.createTextNode(str(b[2])))
        node_x1 = doc.createElement('xmax')
        node_x1.appendChild(doc.createTextNode(str(b[3])))
        node_y1 = doc.createElement('ymax')
        node_y1.appendChild(doc.createTextNode(str(b[4])))
        # node_x2 = doc.createElement('x2')
        # node_x2.appendChild(doc.createTextNode(b['object_box'][4]))
        # node_y2 = doc.createElement('y2')
        # node_y2.appendChild(doc.createTextNode(b['object_box'][5]))
        # node_x3 = doc.createElement('x3')
        # node_x3.appendChild(doc.createTextNode(b['object_box'][6]))
        # node_y3 = doc.createElement('y3')
        # node_y3.appendChild(doc.createTextNode(b['object_box'][7]))
        node_bndbox.appendChild(node_x0)
        node_bndbox.appendChild(node_y0)
        node_bndbox.appendChild(node_x1)
        node_bndbox.appendChild(node_y1)
        # node_bndbox.appendChild(node_x2)
        # node_bndbox.appendChild(node_y2)
        # node_bndbox.appendChild(node_x3)
        # node_bndbox.appendChild(node_y3)

        node_object.appendChild(node_name1)
        node_object.appendChild(node_pose)
        node_object.appendChild(node_truncated)
        node_object.appendChild(node_diffcult)
        node_object.appendChild(node_bndbox)

        root.appendChild(node_object)

    with open(xml_path, 'w') as xf:
        doc.writexml(xf, indent='\t', addindent='\t', newl='\n', encoding="utf-8")

def main():
    label_path ='/home/zoucg/cv_project/data/others/train/others_1024.json'
    xml_dir = '/home/zoucg/cv_project/data/others/train/xml'
    prase_coco(label_path,xml_dir)
    label_path1 ='/home/zoucg/cv_project/data/others/val/others_1024.json'
    xml_dir1 = '/home/zoucg/cv_project/data/others/val/xml'
    prase_coco(label_path1,xml_dir1)

if __name__=='__main__':
    main()