import cv2
import sys
import os
import re
import argparse
import numpy as np
import skimage.io
from skimage.io import imread, imsave
import detect_whole
from models import * 
from utils.utils import *
import shutil
import gdal
from geotif_rw import geotif_read, geotif_write
from geoshp_rw import tif2shp

try:
   from osgeo import gdal
   from osgeo import ogr
   from osgeo import osr
except ImportError:
   import gdal
   import ogr
  
def WriteVectorFile(file_name,all_label,im_geotrans, im_proj):
    # 为了支持中文路径，请添加下面这句代码
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8","NO")
    # 为了使属性表字段支持中文，请添加下面这句
    gdal.SetConfigOption("SHAPE_ENCODING","")

    # strVectorFile =file_name+'shp'
    strVectorFile=file_name

    # 注册所有的驱动
    ogr.RegisterAll()

    # 创建数据，这里以创建ESRI的shp文件为例
    strDriverName = "ESRI Shapefile"
    oDriver =ogr.GetDriverByName(strDriverName)
    if oDriver == None:
        print("%s 驱动不可用！\n", strDriverName)
        return

    # 创建数据源
    oDS =oDriver.CreateDataSource(strVectorFile)
    if oDS == None:
        print("创建文件【%s】失败！", strVectorFile)
        return

    # 创建图层，创建一个多边形图层，这里没有指定空间参考，如果需要的话，需要在这里进行指定
    papszLCO = []
    oLayer =oDS.CreateLayer("TestPolygon", None, ogr.wkbPolygon, papszLCO)
    if oLayer == None:
        print("图层创建失败！\n")
        return

    # 下面创建属性表
    # 先创建一个叫FieldID的整型属性
    oFieldID =ogr.FieldDefn("FieldID", ogr.OFTInteger)
    oLayer.CreateField(oFieldID, 1)

    # 再创建一个叫FeatureName的字符型属性，字符长度为50
    oFieldName =ogr.FieldDefn("FieldName", ogr.OFTString)
    oFieldName.SetWidth(100)
    oLayer.CreateField(oFieldName, 1)

    oDefn = oLayer.GetLayerDefn()
    count = 0
    for a in all_label:
        for i in a:

    # 创建三角形要素
    # oFeatureTriangle = ogr.Feature(oDefn)
    # oFeatureTriangle.SetField(0, 0)
    # oFeatureTriangle.SetField(1, "三角形")
    # geomTriangle =ogr.CreateGeometryFromWkt("POLYGON ((0 0,20 0,10 15,0 0))")
    # oFeatureTriangle.SetGeometry(geomTriangle)
    # oLayer.CreateFeature(oFeatureTriangle)

    # 创建矩形要素
            oFeatureRectangle = ogr.Feature(oDefn)
            oFeatureRectangle.SetField(0, count)

            oFeatureRectangle.SetField(1,i[0].split(' ')[0])
            
            def ImageRowCol2Projection(adfGeoTransform, iCol, iRow):
                dProjX = adfGeoTransform[0] + adfGeoTransform[1] * iCol + adfGeoTransform[2] * iRow
                dProjY = adfGeoTransform[3] + adfGeoTransform[4] * iCol + adfGeoTransform[5] * iRow
                return (dProjX, dProjY)
            x1,y1 = ImageRowCol2Projection(im_geotrans,i[1][0],i[1][1])
            x2,y2 = ImageRowCol2Projection(im_geotrans,i[1][0], i[2][1])
            x3,y3 = ImageRowCol2Projection(im_geotrans,i[2][0] ,i[2][1])
            x4,y4= ImageRowCol2Projection(im_geotrans,i[2][0], i[1][1])
            x5,y5= ImageRowCol2Projection(im_geotrans,i[1][0], i[1][1])
            # xd = "POLYGON (({0} {1},{2} {3},{4} {5},{6} {7},{8} {9}))".fortmat(i[1][0],i[1][1],i[1][0], i[2][1],i[2][0] ,i[2][1],i[2][0], i[1][1],i[1][0], i[1][1])
            xd = "POLYGON (({0} {1},{2} {3},{4} {5},{6} {7},{8} {9}))".format(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5)
            # geomRectangle =ogr.CreateGeometryFromWkt("POLYGON ((30 0,60 0,60 30,30 30,30 0))")
            geomRectangle =ogr.CreateGeometryFromWkt(xd)
            oFeatureRectangle.SetGeometry(geomRectangle)
            oLayer.CreateFeature(oFeatureRectangle)

    # 创建五角形要素
    # oFeaturePentagon = ogr.Feature(oDefn)
    # oFeaturePentagon.SetField(0, 2)
    # oFeaturePentagon.SetField(1, "五角形")
    # geomPentagon =ogr.CreateGeometryFromWkt("POLYGON ((70 0,85 0,90 15,80 30,65 15,700))")
    # oFeaturePentagon.SetGeometry(geomPentagon)
    # oLayer.CreateFeature(oFeaturePentagon)

    oDS.Destroy()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(im_proj)
    # print("shp的投影信息",srs)
    prjFile = open(os.path.splitext(file_name)[0] + ".prj", 'w')
    # 转为字符串
    srs.MorphToESRI()
    prjFile.write(srs.ExportToWkt())
    prjFile.close()
    print("数据集创建完成！\n")

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-3cls.cfg', help='*.cfg path')
parser.add_argument('--names', type=str, default='data/daodan.names', help='*.names path')
# parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
parser.add_argument('--weights', type=str, default='weights/1_or_best.pt', help='weights path')
# parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
parser.add_argument('--source', type=str, default='data/daodan/val', help='source')

parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
opt = parser.parse_args()

device = torch_utils.select_device('1')
model = Darknet(opt.cfg,(512,512))
model.load_state_dict(torch.load(opt.weights)['model'])
model.to(device).eval()


class Param:
    def __init__(self):
        self.img = None
        self.outdir = None
        self.resize_f = '0.0'
        self.ext = None

def save_sub_img(srcImg,param, left, top, img_size_x, img_size_y):
    img = srcImg[top:top + img_size_y, left:left + img_size_x, 0: 3]
    filename = '{}subimg_{}_{}_{}_{}_{}.{}'.format(param.outdir, left, top, img_size_x, img_size_y, param.resize_f, param.ext)
    print('save {}'.format(filename))
    cv2.imwrite(filename, img)
    # imsave(filename, img)
    
def crop_image(img_path,output_dir,img_size, img_overlap=60):
    all_label = []
    import time
    t0 = time.time()
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)
    ext = 'jpg'
    size = 512
    overlap = 30
    # srcImg = imread(image_path)
    # srcImg = cv2.imread(image_path)
    dataset = gdal.Open(image_path)

    gdal_da0 = dataset.ReadAsArray()
    gdal_da1 = gdal_da0.transpose(1,2,0)[:,:,:3]
    gdal_da3 = gdal_da1[:,:,::-1]
    srcImg = gdal_da3
    srcImg = srcImg.copy()
    

    param = Param()
    param.img = srcImg
    param.outdir = output_dir+'/'
    param.ext = ext
    img_width = srcImg.shape[1]
    img_height = srcImg.shape[0]
    #img_width = 450 + 30
    #img_height = 450  +3

    img_size_y = img_size
    img_size_x = img_size
    img_overlap_y = img_overlap
    img_overlap_x = img_overlap
    if img_height < img_size:
        img_size_y = img_height
        img_overlap_y = 0
    if img_width < img_size:
        img_size_x = img_width
        img_overlap_x = 0

    y_cnt = (img_height - img_size_y) // (img_size - img_overlap)
    x_cnt = (img_width - img_size_x) // (img_size - img_overlap)
    print(x_cnt, ' ', y_cnt)

    img_tail_x = img_width - img_size_x - x_cnt * (img_size - img_overlap)
    img_tail_y = img_height - img_size_y - y_cnt * (img_size - img_overlap)
    print(img_tail_x, ' ', img_tail_y)

    y = 0
    last_y = img_tail_y > 0
    last_one = False
    while True:
        top = y * (img_size - img_overlap)
        if last_one:
            top = img_height - img_size
        for x in range(x_cnt + 1):
            left = x * (img_size_x - img_overlap_x)
            # print('{:5}, {:5}, {}, {}'.format(left, top, img_size_x, img_size_y))
            img = srcImg[top:top + img_size_y, left:left + img_size_x, 0: 3]
            filename = '{}subimg_{}_{}_{}_{}_{}.{}'.format(param.outdir, left, top, img_size_x, img_size_y, param.resize_f, param.ext)
            a = detect_whole.detect_whole(img,model,filename,left,top)
            all_label.append(a)
            # save_sub_img(srcImg,param, left, top, img_size_x, img_size_y)
        if img_tail_x > 0:
            print('{:5}, {:5}, {}, {}'.format(img_width - img_size, top, img_size_x, img_size_y))
            # save_sub_img(srcImg,param, left, top, img_size_x, img_size_y)
            img = srcImg[top:top + img_size_y, left:left + img_size_x, 0: 3]
            filename = '{}subimg_{}_{}_{}_{}_{}.{}'.format(param.outdir, left, top, img_size_x, img_size_y, param.resize_f, param.ext)
            a = detect_whole.detect_whole(img,model,filename,left,top)
            all_label.append(a)
        y += 1
        if y > y_cnt:
            if last_y:
                last_y = False
                last_one = True
                continue
            else:
                break

    t2 = time.time()-t0
    print("total time is {}".format(t2))
    return all_label
    # img = srcImg[top:top + img_size_y, left:left + img_size_x, 0: 3]
    # filename = '{}subimg_{}_{}_{}_{}_{}.{}'.format(param.outdir, left, top, img_size_x, img_size_y, param.resize_f, param.ext)
    # detect_whole.detect_whole(img,'weights/best.pt',filename)

def merge_image(image_dir,image_path):
    image_out = image_path
    rects = []
    rect_max = {'left': -1, 'top': -1, 'width': 0, 'height': 0, 'img': ''}
    files = os.listdir(image_dir)
    for line in files:
        line = line.strip('\n')
        m = re.match(r'.*subimg_(\d+)_(\d+)_(\d+)_(\d+)_(\d+(\.\d+)?)\.*', line)
        if m:
            rect = {'left': int(m.group(1)), 'top': int(m.group(2)), 'width': int(m.group(3)), 'height': int(m.group(4)), 'img': line}
            rects.append(rect)
            if rect['left'] >= rect_max['left'] and rect['top'] >= rect_max['top']:
                rect_max = rect

    if rects:
        print('找到{}个图片'.format(len(rects)))
    else:
        print('找不到图片')
        sys.exit(-1)

    img0 = cv2.imdecode(np.fromfile(image_dir + rects[0]['img'], dtype=np.uint8), -1)
    print(img0.shape)
    width = rect_max['left'] + rect_max['width']
    height = rect_max['top'] + rect_max['height']
    shape = (height, width) if len(img0.shape) < 3 else (height, width, img0.shape[2])
    image = np.zeros(shape, dtype=img0.dtype)
    print('输出图片{}x{}'.format(shape[1], shape[0]))

    for rect in rects:
        img = cv2.imdecode(np.fromfile(image_dir + rect['img'], dtype=np.uint8), -1)
        image[rect['top']:rect['top'] + rect['height'], rect['left']:rect['left'] + rect['width']] = img

    print('保存图片: {}'.format(image_out))
    cv2.imencode(os.path.splitext(image_out)[1], image)[1].tofile(image_out)

    # sys.exit(0)

def detect_all(img_dir):
    weights = 'weights/best.pt'
    for root_dir, dir1, path in os.walk(img_dir):
        for p in path:
            if p.endswith('tif'):
                image_path  = os.path.join(root_dir, p)
                img0 = cv2.imread(image_path)
                new_image_name = p[:-4]+ '_'+ image_path.split('/')[-2]+ '.tif'
                new_image_path = os.path.join('/home/zoucg/data/all_data_detect',new_image_name)
                detect_whole(img0,weights,new_image_path)


# def detect_out(image_path,output_dir):
#     if os.path.exists(output_dir) is False:
#         os.makedirs(output_dir)
#     image_name = image_path.split('/')[-1]
#     save_path = os.path.join(output_dir,image_name)
#     temp_dir = './temp/'
#     if os.path.exists(temp_dir) is None:
#         os.makedirs(temp_dir)
#     crop_image(image_path,temp_dir,512)
#     merge_image(temp_dir,save_path)
#     shutil.rmtree(temp_dir)

def detect_out(image_path,output_dir):
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)
    
    geoinfo = geotif_read(image_path)

    image_name = image_path.split('/')[-1]
    save_path = os.path.join(output_dir,image_name)
    temp_dir = './temp/'
    if os.path.exists(temp_dir) is None:
        os.makedirs(temp_dir)
    all_label = crop_image(image_path,temp_dir,512)
    # test_label(image_path,all_label)
    shp_path = save_path[:-3]+'shp'
    WriteVectorFile(shp_path,all_label,geoinfo[2],geoinfo[3])
    merge_image(temp_dir,save_path)


def test_label(image_path, label):
    data = cv2.imread(image_path)
    for a in label:
        for i in a:
            cv2.rectangle(data, i[1], i[2], (225,0,0), -1, cv2.LINE_AA)
    
    cv2.imwrite('test_label.jpg',data)
    # shutil.rmtree(temp_dir)

if __name__=='__main__':
    image_path ='/home/zoucg/cv_project/yolov3/287_kr_shuiyuan_18.tif'
    out = './test_out'
    detect_out(image_path,out)
    # parser = argparse.ArgumentParser()
   
    # # parser.add_argument('--names', type=str, default='./japan6_17.tif', help='image_path')
    # # parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder

    # # opt = parser.parse_args()
    # path ='/home/zoucg/cv_project/yolov3/data/daodan/data_temp/JPEGImages' 
    # for roots,dirs,files in os.walk(path):
    #     for f in files:
    #         if f.endswith('jpg'):
    #             print(f)
    #             image_path = os.path.join(roots,f)
    # # image_path = opt.names
    #             output_dir = opt.output
    #             detect_out(image_path, './aa_s')