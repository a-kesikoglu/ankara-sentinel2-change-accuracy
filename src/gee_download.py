"""Build Sentinel-2 seasonal composites and auxiliary layers in Google Earth Engine
and download them to a common 10 m UTM grid (EPSG:32636) as GeoTIFF.

Outputs (data/processed):
  S2_{year}_features.tif   32 bands, int16 (reflectance x1e4, indices x1e4)
  S2_{year}_obscount.tif   valid observation count, spring and summer
  DW_{year}_label.tif      Dynamic World seasonal mode label (Apr-Sep)
  DW_{year}_prob.tif       Dynamic World mean class probabilities x100 (9 bands, uint8)
  WC_2021.tif              ESA WorldCover v200 map
"""
import io, os, sys, json, yaml
import numpy as np
import ee
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = yaml.safe_load(open(os.path.join(ROOT, 'config', 'config.yaml'), encoding='utf-8'))
OUT = os.path.join(ROOT, 'data', 'processed')
os.makedirs(OUT, exist_ok=True)
ee.Initialize(project=CFG['project']['gee_project'])

BANDS = CFG['sentinel2']['bands']
RES = CFG['project']['pixel_size_m']
CRS = CFG['project']['crs']
TILE = 512

# ---- grid -------------------------------------------------------------------
lon0, lat0, lon1, lat1 = CFG['study_area']['bbox_wgs84']
tr = Transformer.from_crs('EPSG:4326', CRS, always_xy=True)
xs, ys = zip(*[tr.transform(x, y) for x in (lon0, lon1) for y in (lat0, lat1)])
X0 = np.floor(min(xs) / RES) * RES
Y1 = np.ceil(max(ys) / RES) * RES
W = int(np.ceil((max(xs) - X0) / RES))
H = int(np.ceil((Y1 - min(ys)) / RES))
TRANSFORM = from_origin(X0, Y1, RES, RES)
AOI = ee.Geometry.Rectangle([lon0, lat0, lon1, lat1])


def mask_s2(img):
    cp = ee.Image(img.get('cloud_prob')).select('probability')
    scl = img.select('SCL')
    bad = scl.remap(CFG['sentinel2']['scl_masked_classes'],
                    [1] * len(CFG['sentinel2']['scl_masked_classes']), 0)
    ok = cp.lt(CFG['sentinel2']['cloud_probability_threshold']).And(bad.eq(0))
    return img.select(BANDS).updateMask(ok)


def s2_collection(start, end):
    sr = (ee.ImageCollection(CFG['sentinel2']['collection']).filterBounds(AOI)
          .filterDate(start, end)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CFG['sentinel2']['max_scene_cloud_percent'])))
    cp = ee.ImageCollection(CFG['sentinel2']['cloud_probability']).filterBounds(AOI).filterDate(start, end)
    joined = ee.Join.saveFirst('cloud_prob').apply(
        sr, cp, ee.Filter.equals(leftField='system:index', rightField='system:index'))
    return ee.ImageCollection(joined).map(mask_s2)


def add_indices(c, sfx):
    b = lambda n: c.select(n)
    nd = lambda a, d: c.normalizedDifference([a, d])
    bsi = (b('B11').add(b('B4')).subtract(b('B8').add(b('B2')))).divide(
        b('B11').add(b('B4')).add(b('B8')).add(b('B2')))
    idx = ee.Image.cat([nd('B8', 'B4'), nd('B3', 'B8'), nd('B3', 'B11'),
                        nd('B11', 'B8'), nd('B8', 'B11'), bsi]).rename(
        [f'{n}_{sfx}' for n in CFG['sentinel2']['indices']])
    return idx


def features(year_cfg):
    parts, counts, names = [], [], []
    for sfx in ('spring', 'summer'):
        col = s2_collection(*year_cfg[sfx])
        comp = col.median().divide(10000)
        parts.append(comp.rename([f'{b}_{sfx}' for b in BANDS]))
        parts.append(add_indices(comp, sfx))
        counts.append(col.select('B4').count().rename(f'n_{sfx}'))
    feat = ee.Image.cat(parts).multiply(10000).round().toInt16()
    return feat, ee.Image.cat(counts).toInt16()


def grid(x0, y1, w, h):
    return {'dimensions': {'width': w, 'height': h},
            'affineTransform': {'scaleX': RES, 'shearX': 0, 'translateX': x0,
                                'shearY': 0, 'scaleY': -RES, 'translateY': y1},
            'crsCode': CRS}


def download(image, path, dtype, nodata=None):
    if os.path.exists(path):
        print('exists', path); return
    names = image.bandNames().getInfo()
    arr = np.zeros((len(names), H, W), dtype=dtype)
    for r in range(0, H, TILE):
        for c in range(0, W, TILE):
            h, w = min(TILE, H - r), min(TILE, W - c)
            g = grid(X0 + c * RES, Y1 - r * RES, w, h)
            for attempt in range(4):
                try:
                    raw = ee.data.computePixels({'expression': image, 'fileFormat': 'NUMPY_NDARRAY', 'grid': g})
                    break
                except Exception as e:
                    print('retry', r, c, e);
            for i, n in enumerate(names):
                arr[i, r:r + h, c:c + w] = raw[n]
            sys.stdout.write('.'); sys.stdout.flush()
    prof = dict(driver='GTiff', height=H, width=W, count=len(names), dtype=dtype, crs=CRS,
                transform=TRANSFORM, compress='deflate', tiled=True, nodata=nodata)
    with rasterio.open(path, 'w', **prof) as dst:
        dst.write(arr)
        dst.descriptions = tuple(names)
    print('\nwrote', path, arr.shape)


def scene_stats(year_cfg, year):
    out = {}
    for sfx in ('spring', 'summer'):
        col = s2_collection(*year_cfg[sfx])
        out[sfx] = {'n_scenes': col.size().getInfo(),
                    'dates': sorted(set(col.aggregate_array('system:time_start')
                                        .map(lambda t: ee.Date(t).format('YYYY-MM-dd')).getInfo()))}
    return out


if __name__ == '__main__':
    print('grid', W, H, X0, Y1)
    stats = {}
    for key in ('t1', 't2'):
        yc = CFG['periods'][key]; y = yc['year']
        feat, cnt = features(yc)
        download(feat, os.path.join(OUT, f'S2_{y}_features.tif'), 'int16', nodata=-32768)
        download(cnt, os.path.join(OUT, f'S2_{y}_obscount.tif'), 'int16')
        dw = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterBounds(AOI)
              .filterDate(yc['spring'][0], yc['summer'][1]))
        download(dw.select('label').mode().toUint8(), os.path.join(OUT, f'DW_{y}_label.tif'), 'uint8')
        probs = ['water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice']
        download(dw.select(probs).mean().multiply(100).round().toUint8(), os.path.join(OUT, f'DW_{y}_prob.tif'), 'uint8')
        stats[y] = scene_stats(yc, y)
    wc = ee.ImageCollection('ESA/WorldCover/v200').first().toUint8()
    download(wc, os.path.join(OUT, 'WC_2021.tif'), 'uint8')
    json.dump(stats, open(os.path.join(ROOT, 'outputs', 'tables', 'scene_stats.json'), 'w'), indent=1)
    print(json.dumps({y: {s: v['n_scenes'] for s, v in d.items()} for y, d in stats.items()}))
