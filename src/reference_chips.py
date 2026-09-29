"""Build interpretation sheets for the validation sample.

Each row shows one sample point (blind to map labels):
  VHR T1 | S2 T1 summer true colour | S2 T1 spring false colour | VHR T2 | S2 T2 summer true | S2 T2 spring false
VHR = Esri World Imagery Wayback releases (T1: 2019-12-12 release, 2018 imagery; T2: 2026-08-05 release, 2025 imagery).
The central 10 m Sentinel-2 pixel is outlined in every chip.
"""
import os, math, io, time, json
import numpy as np, pandas as pd, requests
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer
from common import *

REL = {'T1': 4756, 'T2': 26334}
META = {'T1': 'World_Imagery_Metadata_2019_r16', 'T2': 'World_Imagery_Metadata_2026_r07'}
Z = 18
HDR = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
       'Referer': 'https://livingatlas.arcgis.com/wayback/'}
CACHE = os.path.join(ROOT, 'data', 'reference', 'wayback_tiles')
SHEETS = os.path.join(ROOT, 'data', 'reference', 'sheets')
os.makedirs(SHEETS, exist_ok=True)
CH = 170          # chip size in pixels
HALF_M = 60       # half window (m) for VHR chips
S2_HALF = 12      # half window (pixels) for Sentinel-2 chips
to_ll = Transformer.from_crs(CFG['project']['crs'], 'EPSG:4326', always_xy=True)


def merc_px(lon, lat):
    n = 256 * 2 ** Z
    return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n


def tile(rid, x, y):
    p = os.path.join(CACHE, str(rid), f'{Z}_{x}_{y}.jpg')
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        for _ in range(4):
            try:
                r = requests.get(f'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/'
                                 f'default028mm/MapServer/tile/{rid}/{Z}/{y}/{x}', headers=HDR, timeout=30)
                if r.status_code == 200:
                    open(p, 'wb').write(r.content); break
                if r.status_code == 404:
                    # no level-18 tile in this release: upsample the quadrant of the level-17 parent
                    r = requests.get(f'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/'
                                     f'default028mm/MapServer/tile/{rid}/{Z - 1}/{y // 2}/{x // 2}', headers=HDR, timeout=30)
                    if r.status_code == 200:
                        par = Image.open(io.BytesIO(r.content)).convert('RGB')
                        qx, qy = (x % 2) * 128, (y % 2) * 128
                        par.crop((qx, qy, qx + 128, qy + 128)).resize((256, 256), Image.BICUBIC).save(p, quality=92)
                    break
            except Exception:
                time.sleep(2)
    try:
        return Image.open(p).convert('RGB')
    except Exception:
        return Image.new('RGB', (256, 256), (80, 80, 80))


def vhr_chip(rid, x, y):
    # window corners in UTM -> lon/lat -> global mercator pixels
    ll = [to_ll.transform(x + dx, y + dy) for dx, dy in ((-HALF_M, HALF_M), (HALF_M, -HALF_M))]
    (px0, py0), (px1, py1) = [merc_px(*p) for p in ll]
    tx0, ty0, tx1, ty1 = int(px0 // 256), int(py0 // 256), int(px1 // 256), int(py1 // 256)
    mos = Image.new('RGB', ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            mos.paste(tile(rid, tx, ty), ((tx - tx0) * 256, (ty - ty0) * 256))
    ox, oy = tx0 * 256, ty0 * 256
    crop = mos.crop((int(px0 - ox), int(py0 - oy), int(px1 - ox), int(py1 - oy))).resize((CH, CH), Image.BILINEAR)
    d = ImageDraw.Draw(crop)
    f = CH / (2 * HALF_M)
    a, b = CH / 2 - 5 * f, CH / 2 + 5 * f
    d.rectangle([a, a, b, b], outline=(255, 255, 0), width=2)
    return crop


def s2_chip(arr, names, r, c, bands, gain):
    ix = [names.index(b) for b in bands]
    r, c = int(r), int(c)
    rr = np.clip(np.arange(r - S2_HALF, r + S2_HALF + 1), 0, arr.shape[1] - 1)
    cc = np.clip(np.arange(c - S2_HALF, c + S2_HALF + 1), 0, arr.shape[2] - 1)
    w = arr[np.ix_(ix, rr, cc)].astype(np.float32) / 10000
    img = np.clip(w.transpose(1, 2, 0) * gain, 0, 1) ** (1 / 1.4)
    im = Image.fromarray((img * 255).astype(np.uint8)).resize((CH, CH), Image.NEAREST)
    d = ImageDraw.Draw(im); f = CH / (2 * S2_HALF + 1)
    d.rectangle([S2_HALF * f, S2_HALF * f, (S2_HALF + 1) * f, (S2_HALF + 1) * f], outline=(255, 255, 0), width=2)
    return im


def capture_dates(sp):
    """Point-in-polygon lookup of VHR source dates from the Wayback metadata services."""
    from shapely.geometry import shape, Point
    out = {}
    lon0, lat0, lon1, lat1 = CFG['study_area']['bbox_wgs84']
    for k, svc in META.items():
        polys = []
        for layer in range(3, 8):
            u = f'https://metadata.maptiles.arcgis.com/arcgis/rest/services/{svc}/MapServer/{layer}/query'
            q = dict(geometry=f'{lon0},{lat0},{lon1},{lat1}', geometryType='esriGeometryEnvelope', inSR=4326, outSR=4326,
                     spatialRel='esriSpatialRelIntersects', outFields='SRC_DATE2,SRC_RES,SRC_ACC,NICE_DESC', f='geojson')
            try:
                g = requests.get(u, params=q, timeout=120).json()
                polys += [(layer, shape(ft['geometry']), ft['properties']) for ft in g.get('features', []) if ft.get('geometry')]
            except Exception as e:
                print('meta fail', k, layer, e)
        dates, res = [], []
        for lo, la in zip(sp.lon, sp.lat):
            pt = Point(lo, la); hit = [p for p in polys if p[1].contains(pt)]
            hit.sort(key=lambda p: -p[0])
            dates.append(hit[0][2].get('SRC_DATE2') if hit else None); res.append(hit[0][2].get('SRC_RES') if hit else None)
        out[k] = (dates, res)
    return out


def needed_tiles(x, y):
    ll = [to_ll.transform(x + dx, y + dy) for dx, dy in ((-HALF_M, HALF_M), (HALF_M, -HALF_M))]
    (px0, py0), (px1, py1) = [merc_px(*p) for p in ll]
    return [(tx, ty) for tx in range(int(px0 // 256), int(px1 // 256) + 1) for ty in range(int(py0 // 256), int(py1 // 256) + 1)]


def prefetch(sp):
    from concurrent.futures import ThreadPoolExecutor
    jobs = {(rid, tx, ty) for x, y in zip(sp.x, sp.y) for rid in REL.values() for tx, ty in needed_tiles(x, y)}
    print('tiles', len(jobs), flush=True)
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda j: tile(*j), jobs))


if __name__ == '__main__':
    sp = pd.read_csv(f'{SAMP}/validation_points.csv')
    prefetch(sp)
    feats = {y: read(f'{PROC}/S2_{y}_features.tif') for y in YEARS}
    try:
        font = ImageFont.truetype('arial.ttf', 14)
    except Exception:
        font = ImageFont.load_default()
    per = 6
    for s in range(0, len(sp), per):
        sub = sp.iloc[s:s + per]
        sheet = Image.new('RGB', (40 + 6 * (CH + 4), 22 + len(sub) * (CH + 4)), 'white')
        d = ImageDraw.Draw(sheet)
        for j, t in enumerate(['VHR 2018', 'S2 2018 summer', 'S2 2018 spring FCC', 'VHR 2025', 'S2 2025 summer', 'S2 2025 spring FCC']):
            d.text((40 + j * (CH + 4) + 4, 3), t, fill='black', font=font)
        for i, (_, p) in enumerate(sub.iterrows()):
            yy = 22 + i * (CH + 4)
            d.text((2, yy + CH // 2 - 7), str(p.pid), fill='black', font=font)
            chips = []
            for k, yr in (('T1', YEARS[0]), ('T2', YEARS[1])):
                arr, _, names = feats[yr]
                chips += [vhr_chip(REL[k], p.x, p.y),
                          s2_chip(arr, names, p.row, p.col, ['B4_summer', 'B3_summer', 'B2_summer'], 3.2),
                          s2_chip(arr, names, p.row, p.col, ['B8_spring', 'B4_spring', 'B3_spring'], 2.4)]
            for j, ch in enumerate(chips):
                sheet.paste(ch, (40 + j * (CH + 4), yy))
        sheet.save(os.path.join(SHEETS, f'sheet_{s // per + 1:03d}.jpg'), quality=88)
        print('sheet', s // per + 1, flush=True)
    cd = capture_dates(sp)
    sp['vhr_date_t1'], sp['vhr_res_t1'] = cd['T1']; sp['vhr_date_t2'], sp['vhr_res_t2'] = cd['T2']
    sp.to_csv(f'{SAMP}/validation_points.csv', index=False)
    print(sp[['vhr_date_t1', 'vhr_date_t2']].value_counts().head(20))
