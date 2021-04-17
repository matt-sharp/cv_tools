# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 11:22:59 2019
@author: Minty
"""
 
import os
import sys
 
# args = sys.argv
# if len(args)!=5:
#     print("Invalid parameters.")
#     print("Please input:")
#     print("Input path of raster 1")
#     print("Input path of raster 2")
#     print("Output path of raster 1")
#     print("Output path of raster 2")
#     sys.exit(1)
 
# print ('参数个数为:', len(args), '个参数。')
# print ('参数列表:', str(args))
 
#input
# in_raster1 = args[0]'
in_raster1 = '/home/zoucg/cv_project/unetplus/guangzhou/20181004.img'
in_raster2 = '/home/zoucg/cv_project/unetplus/guangzhou/20190311.img'
 
out_raster1 = './2018_crop.tif'
out_raster2 = './2019_crop.tif'
# args[3]
 
 
import sys
import gdal
from osgeo import osr
import numpy as np
from gdalconst import *
from osr import SpatialReference
 
def geo2imagexy(dataset, x, y):
    '''
    根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param x: 投影或地理坐标x
    :param y: 投影或地理坐标y
    :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    '''
    trans = dataset.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解
 
def writeTiff(im_data,im_width,im_height,im_bands,im_geotrans,im_proj,path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
 
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape
        #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset
 
#read raster1---------------------------------------------------------------------------------------------------------------
ds1 = gdal.Open(in_raster1,gdal.GA_ReadOnly)
if ds1 is None:
    print ('cannot open ',in_raster1)
    sys.exit(1)
    
gt1 = ds1.GetGeoTransform()
proj1 = ds1.GetProjection()#获取投影信息
# r1 has left, top, right, bottom of dataset's bounds in geospatial coordinates.
r1 = [gt1[0], gt1[3], gt1[0] + (gt1[1] * ds1.RasterXSize), gt1[3] + (gt1[5] * ds1.RasterYSize)]
 
#read raster2------------------------------------------------------------------------------------------------------------
ds2 = gdal.Open(in_raster2,gdal.GA_ReadOnly)
if ds2 is None:
    print ('cannot open ',in_raster2)
    sys.exit(1)
    
gt2 = ds2.GetGeoTransform()
proj2 = ds2.GetProjection()#获取投影信息
# r2 has left, top, right, bottom of dataset's bounds in geospatial coordinates.
r2 = [gt2[0], gt2[3], gt2[0] + (gt2[1] * ds2.RasterXSize), gt2[3] + (gt2[5] * ds2.RasterYSize)]
 
#calculate the intersection area of r1 and r2
intersection = [max(r1[0], r2[0]), min(r1[1], r2[1]), min(r1[2], r2[2]), max(r1[3], r2[3])]
 
#map intersection to pixel intersection
intersection_pixel_r1 = [geo2imagexy(ds1, intersection[0], intersection[1]), geo2imagexy(ds1, intersection[2], intersection[3])]
intersection_pixel_r2 = [geo2imagexy(ds2, intersection[0], intersection[1]), geo2imagexy(ds2, intersection[2], intersection[3])]
 
#read block data
clip_r1 = ds1.ReadAsArray(int(intersection_pixel_r1[0][0]), int(intersection_pixel_r1[0][1]), int(intersection_pixel_r1[1][0])-int(intersection_pixel_r1[0][0]), int(intersection_pixel_r1[1][1])-int(intersection_pixel_r1[0][1]))
clip_r2 = ds2.ReadAsArray(int(intersection_pixel_r2[0][0]), int(intersection_pixel_r2[0][1]), int(intersection_pixel_r2[1][0])-int(intersection_pixel_r2[0][0]), int(intersection_pixel_r2[1][1])-int(intersection_pixel_r2[0][1]))
 
#output clipped raster---------------------------------------------------------------------------------------------------------------
gt_clip_r1 = [intersection[0], gt1[1], gt1[2], intersection[1], gt1[4], gt1[5]]
writeTiff(clip_r1,clip_r1.shape[2],clip_r1.shape[1],clip_r1.shape[0],gt_clip_r1,proj1,out_raster1)
 
gt_clip_r2 = [intersection[0], gt2[1], gt2[2], intersection[1], gt2[4], gt2[5]]
writeTiff(clip_r2,clip_r2.shape[2],clip_r2.shape[1],clip_r2.shape[0],gt_clip_r2,proj2,out_raster2)