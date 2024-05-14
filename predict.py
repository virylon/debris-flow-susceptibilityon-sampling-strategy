import sys
import os
import numpy as np
from osgeo import gdal, gdalconst, ogr
import numpy as np
import scipy.io
import pandas as pd
import math
from net import CNNF
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
# from Model import AONN
from net import CNNF
from datasets import BasicDataset
from torch.utils.data import Dataset
import torch.nn.functional as F

# 'STFANPOINT', 'STFANPOLYG', 'STRUNOUTPOINT','STRUNOUTPOLYGON', 'STSOURCEPOINT', 'STSOURCEPOLYGON'
        
file_path = r'factor\STRUNOUTPOLYGON'
# samples_path = r'samples'
factors = ["aspect", "curvature", "dem", "disfault",
           "disriver", "lithology", "pga", "rain", "slope", "twi"]
# source_polygon_training  source_point_training

gdal.AllRegister()
images = []
for fc in factors:
    fc_path = os.path.join(file_path, fc+'.tif')
    # print(fc_path)
    raster = gdal.Open(fc_path)
    # band = raster.GetRasterBand(1)
    width = raster.RasterXSize  
    height = raster.RasterYSize  
    geotrans = raster.GetGeoTransform()  
    proj = raster.GetProjection() 
    data = raster.ReadAsArray(0, 0, width, height)
    data = np.expand_dims(data, -1)
    images.append(data)

    band = raster.GetRasterBand(1)
    nodata = band.GetNoDataValue()

data[data!=nodata] = 1
data[data==nodata] = 0
mask = data.squeeze(-1)
print(mask.shape)

images = np.concatenate(images, axis=-1)
print(images.shape)
datas = images.reshape(-1, images.shape[-1])
print(datas.shape)

net = CNNF()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device=device)
cp_path = r'checkpoints\STRUNOUTPOLYGON_28_auc_9233.pth'
net.load_state_dict(torch.load(cp_path, map_location=device))


class BasicDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        x = self.data[i]
        x = torch.tensor(x, dtype=torch.float32)
        x = x.unsqueeze(-1)
        x_ = F.normalize(x, dim=0)
        x_ = x_.unsqueeze(-1)

        return x_


dataset = BasicDataset(datas)
loader = DataLoader(dataset, batch_size=4096, shuffle=False,
                    num_workers=0, pin_memory=True)
preds = []
for batch in loader:
    data = batch
    # print(data.shape)
    data = data.to(device=device)
    pred = net(data)
    pred = pred.cpu().detach().numpy()
    # print(pred.shape)
    preds.append(pred)

preds = np.concatenate(preds, axis=0)
pred_image = preds.reshape(images.shape[0], images.shape[1])
print(pred_image.shape)
pred_image = pred_image*mask

drive = gdal.GetDriverByName('GTiff')
name = os.path.split(cp_path)[-1].split('.')[0]
pred_path = os.path.join('predicts',name+'.tif')
out_raster = drive.Create(pred_path,images.shape[1], images.shape[0], 1, gdal.GDT_Float32)
                          
out_raster.SetGeoTransform(geotrans)
out_raster.SetProjection(proj)
band = out_raster.GetRasterBand(1)
band.SetNoDataValue(0)
out_raster.GetRasterBand(1).WriteArray(pred_image)
# out_raster.GetRasterBand(1).WriteArray(resultData)
