import os, yaml
import numpy as np
import rasterio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = yaml.safe_load(open(os.path.join(ROOT, 'config', 'config.yaml'), encoding='utf-8'))
PROC = os.path.join(ROOT, 'data', 'processed')
SAMP = os.path.join(ROOT, 'data', 'samples')
MAPS = os.path.join(ROOT, 'outputs', 'maps')
TABS = os.path.join(ROOT, 'outputs', 'tables')
FIGS = os.path.join(ROOT, 'outputs', 'figures')
for d in (SAMP, MAPS, TABS, FIGS):
    os.makedirs(d, exist_ok=True)

CLASSES = {1: 'Water', 2: 'Tree cover', 3: 'Low vegetation', 4: 'Built-up', 5: 'Bare/sparse'}
COLORS = {1: '#3a7dc9', 2: '#00441b', 3: '#d9ef8b', 4: '#c51b7d', 5: '#f1a340'}
YEARS = (CFG['periods']['t1']['year'], CFG['periods']['t2']['year'])
SEED = CFG['project']['seed']

# Dynamic World label -> study classes (0 = excluded)
DW_MAP = np.array([1, 2, 3, 3, 3, 3, 4, 5, 0], dtype=np.uint8)
# ESA WorldCover code -> study classes
WC_MAP = {10: 2, 20: 3, 30: 3, 40: 3, 50: 4, 60: 5, 70: 0, 80: 1, 90: 3, 95: 3, 100: 3}


def read(path, bands=None):
    with rasterio.open(path) as src:
        arr = src.read(bands) if bands else src.read()
        return arr, src.profile, list(src.descriptions)


def write(path, arr, profile, dtype=None, nodata=None, descriptions=None):
    arr = arr if arr.ndim == 3 else arr[None]
    p = profile.copy()
    p.update(count=arr.shape[0], dtype=dtype or arr.dtype.name, nodata=nodata, compress='deflate')
    with rasterio.open(path, 'w', **p) as dst:
        dst.write(arr.astype(p['dtype']))
        if descriptions:
            dst.descriptions = tuple(descriptions)


def remap_wc(wc):
    out = np.zeros_like(wc, dtype=np.uint8)
    for k, v in WC_MAP.items():
        out[wc == k] = v
    return out
