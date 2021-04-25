import  fiona
try:
    import gdal
except:
    from osgeo import gdal,ogr,osr

import numpy as np

def read_shp(shp_file):
    c = fiona.open(shp_file, 'r')
    ds= []
    for i in c:
        d =i['geometry']['coordinates']
        ds.append(d)
    return np.array(ds)



def change_one(dataset, x,y):
    
   
    trans = dataset.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    return np.linalg.solve(a, b)

    # corad = c['geometry']['coordinates']
    # print(corad)

        


def position_change(tif_file,shp_file,output):
    gdal.AllRegister()
    dataset = gdal.Open(r"/data1/test_airplane/beijing_airport1_2017.tif")
    ps = read_shp(shp_file)
    ps =np.squeeze(ps,1)
    ps = ps[:,:-1,:]
    x = ps[:,:,0]
    y = ps[:,:,1]
    l = ps.shape[0]
    w = ps.shape[1]
    da = np.empty(ps.shape)
    print(ps.shape)
    for i in range(l):
        for j in range(w):
            m,n = change_one(dataset,ps[i,j,0],ps[i,j,1])
            da[i,j,0]= m
            da[i,j,1] =n
    da =da[:,[0,2],]
    # da[:,:,:]
    new_da = da.reshape(l,-1)
    list_da = new_da.tolist()
    return list_da
import random
random.seed(0)

def ff(x):
    conf= random.uniform(0,1.0)
    return f'a {conf} '+ ' '.join(list(map(str,x)))+'\n'

def fff(x):
    return ' '.join(list(map(str,x)))+' airplane 0'+'\n'


def totxt(list_da,out_file):
    list_da = list(map(ff,list_da))
    with open(out_file,'w') as f:
        f.writelines(list_da)






ps = position_change(r"/data1/test_airplane/beijing_airport1_2017.tif",'/data1/test_airplane/beijing_airport1_2017.shp','')
totxt(ps,'airplane1.txt')


def 
print(ps)
# read_shp('/data1/test_airplane/beijing_airport1_2017.shp')
    



