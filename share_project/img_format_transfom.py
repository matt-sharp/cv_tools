import gdal
import cv2
from skimage import io

def gdal_read2cv(im_path):
    # gdal format bgr_opacity
    dataset = gdal.Open(im_path)
    gdal_da0 = dataset.ReadAsArray()
    gdal_da0.
    cv_da = cv2.imread(im_path)
    gdal_da1 = gdal_da0.transpose(1,2,0)[:,:,:3]
    #gdal, rgb
    gdal_da3 = gdal_da1[:,:,::-1]
    # ski ,rgb
    ski_da = io.imread(im_path)[:,:,:3]
    cv_da_u = cv2.imread(im_path,cv2.IMREAD_UNCHANGED)
    a =cv_da-gdal_da3[:]
    aa = gdal_da1-ski_da
    
    return gdal_da

def array2raster(f_name, np_array, driver='GTiff',
                 prototype=None,
                 xsize=None, ysize=None,
                 transform=None, projection=None,
                 dtype=None, nodata=None):
    """
    将ndarray数组写入到文件中
    :param f_name: 文件路径
    :param np_array: ndarray数组
    :param driver: 文件格式驱动
    :param prototype: 文件原型
    :param xsize: 图像的列数
    :param ysize: 图像的行数
    :param transform: GDAL中的空间转换六参数
    :param projection: 数据的投影信息
    :param dtype: 数据存储的类型
    :param nodata: NoData元数据
    """
    # 创建要写入的数据集（这里假设只有一个波段）
    # 分两种情况：一种给定了数据原型，一种没有给定，需要手动指定Transform和Projection
    driver = gdal.GetDriverByName(driver)
    if prototype:
        dataset = driver.CreateCopy(f_name, prototype)
    else:
        if dtype is None:
            dtype = gdal.GDT_Float32
        if xsize is None:
            xsize = np_array.shape[1]  # 数组的列数
        if ysize is None:
            ysize = np_array.shape[0]  # 数组的行数
        dataset = driver.Create(f_name, xsize, ysize, 3, dtype)  # 这里的1指的是一个波段
        dataset.SetGeoTransform(transform)
        dataset.SetProjection(projection)
    # 将array写入文件
    dataset.GetRasterBand(3).WriteArray(np_array)
    if nodata is not None:
        dataset.GetRasterBand(3).SetNoDataValue(nodata)
    dataset.FlushCache()
    return f_name

def test():
        # 打开栅格数据集
    ds = gdal.Open('example.tif') # example.tif有三个波段，分别是蓝，红，近红外

    # 获取数据集的一些信息
    x_size = ds.RasterXSize  # 图像列数
    y_size = ds.RasterYSize  # 图像行数

    proj = ds.GetProjection()  # 返回的是WKT格式的字符串
    trans = ds.GetGeoTransform()  # 返回的是六个参数的tuple

    # 在数据集层面ReadAsArray方法将每个波段都转换为了一个二维数组
    image = ds.ReadAsArray()

    # 获得波段对应的array
    bnd_red = image[1].astype(float)  # 红波段
    bnd_nir = image[2].astype(float)  # 近红外波段

    idx_ndvi = (bnd_nir - bnd_red) / (bnd_nir + bnd_red)  # 计算NDVI指数

    out1_file = 'NDVI.tif'
    array2raster(out1_file, idx_ndvi,
                xsize=x_size, ysize=y_size,
                transform=trans, projection=proj,
                dtype=gdal.GDT_Float32)

    idx_dvi = bnd_nir - bnd_red  # 计算DVI指数

    out2_file = 'DVI.tif'
    # 这里我们使用out1_file作为原型图像作为参考来保存out2_file
    array2raster(out2_file, idx_ndvi, prototype=gdal.Open(out1_file))

    # 关闭数据集
    ds = None

def main():
    im_path = "/home/zoucg/data/dada_down/japanese12_2008130915/18/japanese12_2008130915.tif"
    # im_path ='/home/zoucg/cv_project/yolov3/data/daodan/data_temp/JPEGImages/japan1_20161231_16__768__384.jpg'
    da = gdal_read2cv(im_path)

if __name__=='__main__':
    main()