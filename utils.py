import math
import numpy as np

CDELT = 0.599733;
HPCCENTER = 4096.0 / 2.0;
rsun_meters = 696000;
dsun_meters = 149600000;
DEFAULT_WIDTH, DEFAULT_HEIGHT = 64, 64
IMAGE_WIDTH, IMAGE_HEIGHT = 512, 512

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def LoadNormalizedData(path):
    tmp = np.load(path)['x']
    tmp = NormalizeData(tmp)
    
    return tmp

def LoadData(path):
    tmp = np.load(path)['x']
    
    return tmp

def ConvertHPCToPixXY(hpc_coord):
    x = HPCCENTER + (hpc_coord[0] / CDELT)
    y = HPCCENTER - (hpc_coord[1] / CDELT)
    
    return (x, y)

def ResizeCoord(coord, original=4096, new_size=IMAGE_WIDTH):
    factor = original // new_size
    
    return (coord[0]//factor, coord[1]//factor)

def GetClosestMultipleDown(num, factor):
    if num < factor:
        return 0
    
    return num - (num % factor)

def GetClosestMultipleUp(num, factor):
    if num < factor:
        return factor
    
    return num + factor - (num % factor)

def Image2DToDict(im):
    d = {}
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            d[(i, j)] = im[i][j]
    return d

def PixelDistance(coord1, coord2):
    return math.sqrt((coord2[0]-coord1[0])**2 + (coord2[1]-coord1[1])**2)

def PadMatrixWithValue(matrix, val=0, size=20):
    mat = matrix
    w, h = matrix.shape[0], matrix.shape[1]
    mat[:size,:] = mat[:,:size] = mat[:,w-size:] =  mat[h-size:,:] = val
    
    return mat