import numpy as np

from utils import *

DEFAULT_WIDTH, DEFAULT_HEIGHT = 64, 64
IMAGE_WIDTH, IMAGE_HEIGHT = 512, 512
NMS_KERNEL_SIZE = 4

def GetTopNDistancedPoints(im, n=3, distance_threshold=15):
    im_dict = Image2DToDict(im)
    distanced_points = set()
    im_dict = {k: v for k, v in sorted(im_dict.items(), key=lambda item: item[1], reverse=True)}
    counter = 0
    
    for k, v in im_dict.items():
        if counter >= n:
            break
        if not distanced_points:
            distanced_points.add(k)
            counter += 1
            continue
        above_threshold = True
        for p in distanced_points:
            dist = PixelDistance(k, p)
            if dist < distance_threshold:
                above_threshold = False
                break
        if above_threshold:
            distanced_points.add(k)
            counter += 1
    
    return distanced_points

def NMSImage(im, k=NMS_KERNEL_SIZE):
    sums = []
    new_image = []
    
    for i in range(0, im.shape[0], k):
        tmp = []
        for j in range(0, im.shape[1], k):
            coord = NMSKernel(im, (i, j), k)
            tmp.append(im[coord[0]][coord[1]])
        new_image.append(tmp)
        
    return np.array(new_image)

def NMSKernel(im, coord, k):
    x, y = coord[0], coord[1]
    max_val = 0
    max_val_coord = coord
    
    for i in range(-k, k):
        for j in range(-k, k):
            try:
                if im[x+i][y+j] >= max_val:
                    max_val = im[x+i][y+j]
                    max_val_coord = (x+i, y+j)
            except IndexError:
                continue
                
    return max_val_coord

def GetImageTopNRegionsCutouts(im, N=3):
    upsample_factor = NMS_KERNEL_SIZE
    im = PadMatrixWithValue(im)
    nms_im = NMSImage(im)
    top_n_points = list(GetTopNDistancedPoints(nms_im, N))
    top_n_points = [[x[0]*upsample_factor, x[1]*upsample_factor] for x in top_n_points]
    
    return GetCoordCutouts(im, top_n_points)

def GetCoordCutouts(im, coords):
    cutouts = []
    for coord in coords:
        cutouts.append(GetCutout(im, coord))

    return cutouts

def GetCutout(im, coord, N=DEFAULT_WIDTH):
    x_start = int(coord[0]-N//2)
    x_end = int(coord[0]+N//2)
    y_start = int(coord[1]-N//2)
    y_end = int(coord[1]+N//2)
    cutout_array = im[x_start:x_end, y_start:y_end]
    
    return cutout_array