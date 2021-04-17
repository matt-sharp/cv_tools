# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import sys
import io
import copy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
try:
    from osgeo import gdal
    from osgeo import ogr
    from osgeo import osr
except ImportError:
    import gdal
    import ogr
    import osr


def ImageRowCol2Projection(adfGeoTransform, iCol, iRow):
    dProjX = adfGeoTransform[0] + adfGeoTransform[1] * iCol + adfGeoTransform[2] * iRow
    dProjY = adfGeoTransform[3] + adfGeoTransform[4] * iCol + adfGeoTransform[5] * iRow
    return (dProjX, dProjY)

def tif2shp(result_path_name, all_label, im_geotrans, im_proj):
    # im_data = np.where(im_data == 1, 1, 0).astype(np.uint8)
    ogr.RegisterAll()
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES");
    gdal.SetConfigOption("SHAPE_ENCODING", "CP936");
    # gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
    # gdal.SetConfigOption("SHAPE_ENCODING", "")
    # ogr.RegisterAll()
    # ogr.RegisterAll();
    # gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES");
    # gdal.SetConfigOption("SHAPE_ENCODING", "CP936");
    strDriverName = "ESRI Shapefile"
    oDriver = ogr.GetDriverByName(strDriverName)
    if oDriver == None:
        print("%s 驱动不可用！\n", strDriverName)
    oDS = oDriver.CreateDataSource(result_path_name)
    if oDS == None:
        print("创建文件【%s】失败！", result_path_name)

    # srs = osr.SpatialReference()
    # srs.ImportFromEPSG(4326)
    papszLCO = []
    oLayer = oDS.CreateLayer("TestPolygon", None, ogr.wkbPolygon, papszLCO)
    if oLayer == None:
        print("图层创建失败！\n")
    oFieldID = ogr.FieldDefn("FieldID", ogr.OFTInteger)
    oLayer.CreateField(oFieldID, 1)
    oDefn = oLayer.GetLayerDefn()

    contours, hierarchy = cv2.findContours(im_data, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    relat = hierarchy[0]
    rootor = relat[:,-1] == -1
    root = [ a for a in range(len(rootor)) if rootor[a] == 1]
    seeds2 = root
    def first_seek(root):
        seed0 = []
        seed1 = []
        for i in root:
            firstor = relat[:,-1] ==  i
            first = [ a for a in range(len(firstor)) if firstor[a] == 1]
            first_copy=copy.copy(first)
            seed0.append(first_copy)
            first.append(i)
            seed1.append(first)
        seed2 = [i for item in seed0 for i in item]
        return seed1,seed2
        
    seeds_all=[]
    for i in range(10):
        if seeds2:
            seeds1,seeds2 = first_seek(seeds2)
            if  (i % 2) == 0:
                for j in seeds1:
                    seeds_all.append(j)
    '''上面代码为根据轮廓层次结构判定各层父子关系'''

    for j in seeds_all:
        gardens = ogr.Geometry(ogr.wkbMultiPolygon) #定义总的多边形集
        for k in j:
            area = cv2.contourArea(contours[k])
        
            if area > 10:#面积大于n才保存
                box1 = ogr.Geometry(ogr.wkbLinearRing)
                for point in contours[k]:
                    x_col = float(point[0,1])
                    y_row = float(point[0,0])
                    coordinate_x,coordinate_y = ImageRowCol2Projection(im_geotrans, y_row, x_col)
                    box1.AddPoint(coordinate_x,coordinate_y)
                oFeatureTriangle = ogr.Feature(oDefn)
                oFeatureTriangle.SetField(0, 1)
                garden1 = ogr.Geometry(ogr.wkbPolygon) #每次重新定义单多变形
                garden1.AddGeometry(box1)       #将轮廓坐标放在单多边形中
                gardens.AddGeometry(garden1)    #依次将单多边形放入总的多边形集中
        if gardens.IsEmpty():
            continue
        gardens.CloseRings() #封闭多边形集中的每个单多边形，后面格式需要
        geomTriangle =ogr.CreateGeometryFromWkt(str(gardens)) #将封闭后的多边形集添加到属性表
        oFeatureTriangle.SetGeometry(geomTriangle)
        oLayer.CreateFeature(oFeatureTriangle)

    oDS.Destroy()
    # ------------------------------------设置投影信息---------------------------
    srs = osr.SpatialReference()
    srs.ImportFromWkt(im_proj)
    # print("shp的投影信息",srs)
    prjFile = open(os.path.splitext(result_path_name)[0] + ".prj", 'w')
    # 转为字符串
    srs.MorphToESRI()
    prjFile.write(srs.ExportToWkt())
    prjFile.close()
    # print("数据集创建完成！\n")