import sys
import os
import numpy as np
from osgeo import gdal, gdalconst, ogr
import numpy as np
import scipy.io
import pandas as pd
import math


# os.chdir(r"samples")
# samples_path = r'samples'
# source_polygon_training   source_point_training
# source_point   source_polygon
# stack_point  stack_polygon
# flow_point  flow_polygon
path = r'factor'
keys = ['STFANPOINT', 'STFANPOLYG', 'STRUNOUTPOINT',
        'STRUNOUTPOLYGON', 'STSOURCEPOINT', 'STSOURCEPOLYGON']
for key in keys:
    factors = [key, "aspect", "curvature", "dem", "disfault",
               "disriver", "lithology", "pga", "rain", "slope", "twi"]
    columns = ["label", "aspect", "curvature", "dem", "disfault",
               "disriver", "lithology", "pga", "rain", "slope", "twi"]

    # 获取注册类
    gdal.AllRegister()
    datas = []
    file_path = os.path.join(path, key)
    for fc in factors:
        fc_path = os.path.join(file_path, fc+'.tif')
        print(fc_path)
        raster = gdal.Open(fc_path)
        # band = raster.GetRasterBand(1)
        width = raster.RasterXSize  
        height = raster.RasterYSize  
        geotrans = raster.GetGeoTransform() 
        proj = raster.GetProjection()  
        data = raster.ReadAsArray(0, 0, width, height)
        data = np.expand_dims(data, -1)
        datas.append(data)
        # scipy.io.savemat(os.path.join(samples_path, "factors.mat"), dic)
    datas = np.concatenate(datas, axis=-1)
    datas = datas.reshape(-1, datas.shape[-1])
    print(datas.shape)

    df = pd.DataFrame(columns=columns, data=datas)
    df.drop(df[(df.values == -32768)].index, inplace=True)
    df.drop(df[(df.label == 3)].index, inplace=True)
    df_path = os.path.join('csv', key+'.csv')
    df.to_csv(df_path)
