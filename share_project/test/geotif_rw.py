import gdal
#---------------------------gzs------------------
import sys,os
try:
    from osgeo import gdal
    from osgeo import ogr
    from osgeo import osr
except ImportError:
    import gdal
    import ogr
    import osr
#-------------------------end------------------------


def geotif_read(root_path):
    dataset = gdal.Open(root_path)
    width = dataset.RasterXSize         # 获取数据宽度
    height = dataset.RasterYSize        # 获取数据高度
    # outbandsize = dataset.RasterCount   # 获取数据波段数
    im_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
    im_proj = dataset.GetProjection()  # 获取投影信息
    # datatype = dataset.GetRasterBand(1).DataType
    # im_data = dataset.ReadAsArray()
    # return [width, height, im_data, im_geotrans, im_proj]
    return [width, height, im_geotrans, im_proj]


def geotif_write(result_path_name, im_data, im_geotrans, im_proj, width, height, outbandsize=1):
    # 数据保存
    # base_name,ext=os.path.splitext(os.path.basename(result_path_name))
    driver = gdal.GetDriverByName("GTiff")
    outdataset = driver.Create(result_path_name, width, height, 1, 1) #result_path_name
    if (outdataset != None):
        outdataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        outdataset.SetProjection(im_proj)  # 写入投影
    for i in range(outbandsize):
        outdataset.GetRasterBand(i + 1).WriteArray(im_data)


    del outdataset
if __name__=='__main__':
    geotif_read('/home/zoucg/Downloads/qq-files/498072518/file_recv/287_kr_shuiyuan_18.tif')
