import os
from pathlib import Path
import sys

from osgeo import gdal, osr, ogr


def raster2shp(classifiy_raster, output_dir, bkg_value=0):
    ds = gdal.Open(classifiy_raster, gdal.GA_ReadOnly)
    srcband = ds.GetRasterBand(1)
    maskband = srcband.GetMaskBand()

    out_shp = os.path.join(output_dir, f'{Path(classifiy_raster).stem}.shp')
    drv = ogr.GetDriverByName('ESRI Shapefile')
    dst_ds = drv.CreateDataSource(out_shp)
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromWkt(ds.GetProjection())

    dst_layername = 'out'
    dst_layer = dst_ds.CreateLayer(dst_layername, srs=dst_srs)
    dst_fieldname = 'Category'
    fd = ogr.FieldDefn(dst_fieldname, ogr.OFTInteger)
    dst_layer.CreateField(fd)
    dst_field = 0
    options = []
    # 参数：输入栅格图像波段、掩码图像波段、矢量化后的矢量图层、需要将值写入属性字段的索引,比如Category索引为0、算法选项、进度条回调函数、进度条参数
    gdal.Polygonize(srcband, maskband, dst_layer, dst_field, options)

    ds = None

    # Remove backgroud value from converted shapefile
    pFeaturelayer = dst_ds.GetLayer(0)
    strFilter = f"Category = '{bkg_value}'"
    pFeaturelayer.SetAttributeFilter(strFilter)
    for pFeature in pFeaturelayer:
        pFeatureFID = pFeature.GetFID()
        pFeaturelayer.DeleteFeature(int(pFeatureFID))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,default=r'C:\Users\EDZ\Desktop\building_regulation\demo_image\TEST_1m2_classify.tif',
                        #default='F:/hongtu-test/change_detection/unetplus/img4-6_mask.tif',
                        help='input raster to be vectorized')
    print('a')
    parser.add_argument('output_dir',
                        type=str,default=r'C:\Users\EDZ\Desktop\building_regulation\results',
                        #default=r"F:\hongtu-test\change_detection\unetplus\test",
                        help='output directory for shapefile')
    parser.add_argument(
        '--background',
        type=int,
        default=0,
        help=
        'backgroud value in raster image which will be excluded in the converted shapefile.'
    )

    # args = parser.parse_args()
    # print(args)
    image_file = r'C:\Users\EDZ\Desktop\building_regulation\demo_image\TEST_1m2_classify.tif'
    out_dir = r'C:\Users\EDZ\Desktop\building_regulation\results'

    raster2shp(image_file, out_dir , 0)
