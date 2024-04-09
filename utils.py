import argparse
import heapq
import os
import sys
import time
from math import asin, cos, radians, sin, sqrt
from typing import Dict, Union

import geopandas as gpd
import loguru
import numpy as np
import rasterio as rio
import torch as th
import yaml
from matplotlib import pyplot as plt
from rasterio.features import geometry_mask
from scipy import sparse as sp
from PIL import Image
import bottleneck as bn
import json


def read_config(path: str, silence: bool=False) -> Union[str, Dict]:
    config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)
    segments = path.split('.')
    for s in segments:
        config = config[s]
    not silence and logger.debug(f'Read config: {path} = {config}')
    return config

logger = loguru.logger
logger_format = '<green>{time:YYYY-MM-DD HH:mm:ss.SS}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>'
logger_config = {
    'handlers': [{
        'sink': sys.stdout,
        'format': logger_format,
        'enqueue': True, # Multiprocessing-safe
    }, {
        'sink': 'logs/logs_{time:YYYYMMDD}.log',
        'format': logger_format,
        'rotation': '1 day',
        'enqueue': True,
    }]
}

logger.configure(**logger_config)

device = th.device(read_config('device'))

class Scaler(object):
    def __init__(self, data: np.ndarray):
        # self.max = data.max()
        self.max = np.percentile(data, 99.5)
        self.min = data.min()
        logger.debug(f'Scaler loaded. max={self.max}, min={self.min}')

    def norm(self, data: np.ndarray):
        # return (data - self.min) / (self.max - self.min)
        return data / self.max

    def inv(self, data: np.ndarray):
        return data * self.max

class ArgparseFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass

def astensor(*args: np.ndarray):
    """
    Accept one or multiple numpy array, return the same number of tensors moved
        to device specified in config.yaml.
    """

    def _astensor(a):
        if isinstance(a, th.Tensor):
            return a
        return th.tensor(a).to(device).to(th.float32)
    if len(args) == 1:
        return _astensor(args[0])
    else:
        return tuple(_astensor(a) for a in args)

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
        on the earth (specified in decimal degrees) in **meters**
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # radius of earth in km
    return c * r * 1000

def calc_random_walk(A: np.ndarray):
    """Random walk matrix for Diffusion Graph Convolution.
    M = 1/D * A, where D is the diagonal matrix of A.

    Args:
        A (np.ndarray): Adjacency matrix.
    """
    D = np.sum(A, axis=1)
    D_inv = np.reciprocal(D, where=D != 0)
    D_mat_inv = np.diag(D_inv)
    M = np.dot(D_mat_inv, A)
    return M

def outdir(dir: str) -> str:
    """
    Create directory of for the desired output file if needed.
    """
    basefolder = os.path.dirname(dir)
    if not os.path.exists(basefolder):
        os.makedirs(basefolder, exist_ok=True)
    return dir

def datestr() -> str:
    return time.strftime('%Y%m%d%H%M')[2:]


def mse(pred: Union[np.ndarray, th.Tensor], truth: Union[np.ndarray, th.Tensor], squared: bool = True):
    """RMSE ignoring Nan"""
    if isinstance(pred, np.ndarray):
        mse = np.nanmean((pred - truth)**2)
        if not squared:
            return np.sqrt(mse)
        else:
            return mse
    elif isinstance(pred, th.Tensor):
        mse = th.nanmean((pred - truth)**2)
        if not squared:
            return th.sqrt(mse)
        else:
            return mse
    else:
        raise ValueError('Unknown type for mse')


def ssim(pred: Union[np.ndarray, th.Tensor], truth: Union[np.ndarray, th.Tensor]):
    """Calculate Structural Similarity on the floating numbers, ignoring NaN"""
    if isinstance(pred, th.Tensor):
        pred = asnumpy(pred)
        truth = asnumpy(truth)
    # set nan to 0
    pred = pred.copy()
    truth = truth.copy()
    pred[np.isnan(pred)] = 0
    truth[np.isnan(truth)] = 0
    pred = pred / pred.max()
    truth = truth / truth.max()
    return structural_similarity(pred, truth, data_range=1)

def region_mask(shp: gpd.GeoDataFrame, grid: np.ndarray) -> np.ndarray:
    """Generate a 0-1 numpy array mask from a shapefile.

    Args:
        shp (gpd.GeoDataFrame): Shapefile gpd frame.
        grid (np.ndarray): Grid used for generating mask. Only the size (height, width) will be used.

    Returns:
        np.ndarray: The 0-1 mask where 1 denotes land and 0 denotes ocean of shape [h, w]:
        mask[0, 0] is the lower-left corner.

        h_max, w_min           h_max, w_max
            ┌─────────────────────────┐
            │   region_mask (h, w)    │
            └─────────────────────────┘
        h_min, w_min            h_min, w_max
    """
    mask = np.zeros_like(grid, dtype=bool)
    xmin, ymin, xmax, ymax = shp.total_bounds
    mask_bool = geometry_mask(
        shp.geometry,
        out_shape=mask.shape,
        transform=rio.transform.from_bounds(xmin, ymax, xmax, ymin, mask.shape[1], mask.shape[0]),
        all_touched=True)
    mask = 1 - mask_bool.astype(int)
    return mask

def adj_top_k(A: np.ndarray, k: int, largest: bool = True):
    """
    Get the adjacency matrix that only the top K nearest neighbors for each node
        are kept. (smaller distance == larger weight)
        (or keep the nsmallest when largest = False)
    """
    A = A.copy()
    # np.fill_diagonal(A, 0)
    for i in range(A.shape[0]):
        row = A[i, :]
        if largest:
            topk = heapq.nlargest(k, row)[-1]
            A[i, :][A[i, :]<topk] = 0
        else:
            topk = heapq.nsmallest(k, row)[-1]
            A[i, :][A[i, :]>topk] = 0
    return A

def np2png(data: np.ndarray, outdir: str):
    """2d numpy data to RGBA png.
        max value = white (255,255,255,255), min = black (0,0,0,255),
        nan = transparent (0,0,0,0)

    Args:
        data (np.ndarray): [h x w]
        outpath (str): Output path
    """
    if len(data.shape) != 2:
        logger.error(f'cannot save numpy data of shape {data.shape} to png.')
        return
    image_data = (data / bn.nanmax(data) * 255).astype(np.uint8)
    alpha = np.where(np.isnan(data), 0, 255).astype(np.uint8)
    image_data = np.where(np.isnan(data), 0, image_data)
    image_array = np.stack([image_data, image_data, image_data, alpha], axis=-1)
    Image.fromarray(image_array).save(outdir)

def np2json(data: np.ndarray, outdir: str, decimal: int=4, with_png: bool=True):
    """Depcrated - output file too large.
    """
    rounded_data = np.round(data, decimal)
    np.savetxt(outdir, rounded_data, delimiter=',', fmt='%.' + str(decimal) + 'f')
    if with_png:
        np2png(data, outdir + '.png')

def npsave(data: np.ndarray, path: str, with_png: bool=True, dtype=np.float32):
    data = data.astype(dtype)
    with open(outdir(path), 'wb') as f:
        np.save(f, data)
    if with_png:
        np2png(data, path + '.png')

def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.debug(f'Executing {func.__name__} took {time.time() - start:.4f}s.')
        return result
    return wrapper


class NpEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

if __name__ == '__main__':
    # mask = region_mask(gpd.read_file('./data/ushcn/terrain.shp'), np.zeros((1000, 2000)))
    # plt.imshow(mask)
    # main()
    A = np.random.rand(20, 20)
    A[A < 0.5] = np.nan
    np2png(A, './a.png')
    # print(adj_top_k(A, 3))
